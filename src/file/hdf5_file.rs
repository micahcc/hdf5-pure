use crate::error::Error;
use crate::error::Result;
use crate::file::group::Group;
use crate::file::helpers::read_offset_from_slice;
use crate::file::node::Node;
use crate::io::ReadAt;
use crate::object_header::ObjectHeader;
use crate::object_header::messages::MessageType;
use crate::superblock::Superblock;

/// An opened HDF5 file.
///
/// This is the main entry point for reading HDF5 files. It holds a reference
/// to the underlying reader and the parsed superblock.
pub struct File<R: ReadAt + ?Sized> {
    pub(crate) reader: Box<R>,
    pub(crate) superblock: Superblock,
}

impl File<[u8]> {
    /// Open an HDF5 file from an in-memory byte buffer.
    pub fn from_bytes(data: Box<[u8]>) -> Result<File<[u8]>> {
        let superblock = Superblock::parse(&*data, 0)?;
        Ok(File {
            reader: data,
            superblock,
        })
    }
}

impl<R: ReadAt> File<R> {
    /// Open an HDF5 file from any `ReadAt` implementation.
    pub fn from_reader(reader: R) -> Result<File<R>> {
        let superblock = Superblock::parse(&reader, 0)?;
        Ok(File {
            reader: Box::new(reader),
            superblock,
        })
    }
}

impl File<std::fs::File> {
    /// Open an HDF5 file from a filesystem path.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<File<std::fs::File>> {
        let f = std::fs::File::open(path).map_err(Error::Io)?;
        let superblock = Superblock::parse(&f, 0)?;
        Ok(File {
            reader: Box::new(f),
            superblock,
        })
    }
}

impl<R: ReadAt + ?Sized> File<R> {
    /// The parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Access the root group.
    pub fn root_group(&self) -> Result<Group<'_, R>> {
        let addr = self.superblock.root_group_object_header_address;
        let header = ObjectHeader::parse(
            &*self.reader,
            addr,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )?;
        Ok(Group {
            file: self,
            address: addr,
            header,
        })
    }

    /// Open a path like `"/group1/subgroup/dataset"`.
    ///
    /// Returns a `Node` which can be either a `Group` or a `Dataset`.
    pub fn open_path(&self, path: &str) -> Result<Node<'_, R>> {
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        let root = self.root_group()?;

        if parts.is_empty() {
            return Ok(Node::Group(root));
        }

        let mut current_group = root;

        for (i, part) in parts.iter().enumerate() {
            let is_last = i == parts.len() - 1;
            let link = current_group.find_link(part)?;

            match link.target {
                crate::link::LinkTarget::Hard { address } => {
                    let header = ObjectHeader::parse(
                        &*self.reader,
                        address,
                        self.superblock.size_of_offsets,
                        self.superblock.size_of_lengths,
                    )?;

                    if is_last {
                        // Determine if this is a group or dataset
                        let has_layout = header
                            .messages
                            .iter()
                            .any(|m| m.msg_type == MessageType::DataLayout);
                        if has_layout {
                            return Ok(Node::Dataset(crate::file::dataset::Dataset {
                                file: self,
                                address,
                                header,
                            }));
                        } else {
                            return Ok(Node::Group(Group {
                                file: self,
                                address,
                                header,
                            }));
                        }
                    } else {
                        current_group = Group {
                            file: self,
                            address,
                            header,
                        };
                    }
                }
                crate::link::LinkTarget::Soft { ref path } => {
                    // Resolve soft link by re-opening from root
                    let resolved = self.open_path(path)?;
                    if is_last {
                        return Ok(resolved);
                    }
                    match resolved {
                        Node::Group(g) => current_group = g,
                        Node::Dataset(_) => {
                            return Err(Error::NotAGroup {
                                path: part.to_string(),
                            });
                        }
                    }
                }
                crate::link::LinkTarget::External { .. } => {
                    return Err(Error::Other {
                        msg: "external links not supported".into(),
                    });
                }
            }
        }

        Ok(Node::Group(current_group))
    }

    pub(crate) fn size_of_offsets(&self) -> u8 {
        self.superblock.size_of_offsets
    }

    pub(crate) fn size_of_lengths(&self) -> u8 {
        self.superblock.size_of_lengths
    }

    /// Resolve a shared message record to the actual message data.
    ///
    /// When a message has the "shared" flag set, its body is a shared message
    /// record pointing elsewhere — typically a committed datatype. This reads
    /// the object header at the target address and extracts the real message.
    pub(crate) fn resolve_shared_message(
        &self,
        data: &[u8],
        expected_type: MessageType,
    ) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Err(Error::InvalidObjectHeader {
                msg: "empty shared message record".into(),
            });
        }

        let so = self.size_of_offsets();
        let o = so as usize;
        let version = data[0];

        let addr = match version {
            1 => {
                // v1: flags(1) + reserved(6) + address(O)
                if data.len() < 8 + o {
                    return Err(Error::InvalidObjectHeader {
                        msg: "shared message v1 record too short".into(),
                    });
                }
                read_offset_from_slice(data, 8, so)
            }
            2 => {
                // v2: type(1) + address(O) [type 0] or heap_id [type 1]
                if data.len() < 2 {
                    return Err(Error::InvalidObjectHeader {
                        msg: "shared message v2 record too short".into(),
                    });
                }
                let stype = data[1];
                if stype == 1 {
                    return Err(Error::Other {
                        msg: "shared object header message (SOHM) heap lookup not supported".into(),
                    });
                }
                if data.len() < 2 + o {
                    return Err(Error::InvalidObjectHeader {
                        msg: "shared message v2 record too short for address".into(),
                    });
                }
                read_offset_from_slice(data, 2, so)
            }
            3 => {
                // v3: type(1) + address(O) [types 0/2] or heap_id(8) [type 1]
                if data.len() < 2 {
                    return Err(Error::InvalidObjectHeader {
                        msg: "shared message v3 record too short".into(),
                    });
                }
                let stype = data[1];
                match stype {
                    0 | 2 => {
                        // 0 = shared in another object header
                        // 2 = committed message (named datatype)
                        if data.len() < 2 + o {
                            return Err(Error::InvalidObjectHeader {
                                msg: "shared message v3 record too short for address".into(),
                            });
                        }
                        read_offset_from_slice(data, 2, so)
                    }
                    1 => {
                        return Err(Error::Other {
                            msg: "shared object header message (SOHM) heap lookup not supported"
                                .into(),
                        });
                    }
                    _ => {
                        return Err(Error::InvalidObjectHeader {
                            msg: format!("unknown shared message type {}", stype),
                        });
                    }
                }
            }
            _ => {
                return Err(Error::InvalidObjectHeader {
                    msg: format!("unsupported shared message version {}", version),
                });
            }
        };

        if addr == u64::MAX {
            return Err(Error::InvalidObjectHeader {
                msg: "shared message points to undefined address".into(),
            });
        }

        // Read the object header at the target address and find the matching message
        let header = ObjectHeader::parse(
            &*self.reader,
            addr,
            self.size_of_offsets(),
            self.size_of_lengths(),
        )?;

        for msg in &header.messages {
            if msg.msg_type == expected_type {
                return Ok(msg.data.clone());
            }
        }

        Err(Error::InvalidObjectHeader {
            msg: format!(
                "shared message target at {:#x} has no {:?} message",
                addr, expected_type
            ),
        })
    }
}
