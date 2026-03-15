use crate::btree2::BTree2Header;
use crate::btree2::{self};
use crate::error::Error;
use crate::error::Result;
use crate::file::attribute::Attribute;
use crate::file::attribute::parse_attributes;
use crate::file::attribute::parse_attributes_by_creation_order;
use crate::file::dataset::Dataset;
use crate::file::hdf5_file::File;
use crate::file::helpers::read_offset_from_slice;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::{self};
use crate::io::ReadAt;
use crate::link::Link;
use crate::link::LinkTarget;
use crate::object_header::ObjectHeader;
use crate::object_header::messages::MessageType;

/// A group (directory-like container) in the HDF5 file.
pub struct Group<'a, R: ReadAt + ?Sized> {
    pub(crate) file: &'a File<R>,
    #[allow(dead_code)]
    pub(crate) address: u64,
    pub(crate) header: ObjectHeader,
}

impl<'a, R: ReadAt + ?Sized> Group<'a, R> {
    /// List all child link names.
    pub fn members(&self) -> Result<Vec<String>> {
        let links = self.read_links()?;
        Ok(links.into_iter().map(|l| l.name).collect())
    }

    /// List all child link names in creation order.
    ///
    /// Falls back to name order if creation order tracking is not available.
    pub fn members_by_creation_order(&self) -> Result<Vec<String>> {
        let links = self.read_links_by_creation_order()?;
        Ok(links.into_iter().map(|l| l.name).collect())
    }

    /// Get a specific child by name, returning the link.
    pub fn find_link(&self, name: &str) -> Result<Link> {
        let links = self.read_links()?;
        links
            .into_iter()
            .find(|l| l.name == name)
            .ok_or_else(|| Error::PathNotFound {
                path: name.to_string(),
            })
    }

    /// Open a child group by name.
    pub fn group(&self, name: &str) -> Result<Group<'a, R>> {
        let link = self.find_link(name)?;
        match link.target {
            LinkTarget::Hard { address } => {
                let header = ObjectHeader::parse(
                    &*self.file.reader,
                    address,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                )?;
                Ok(Group {
                    file: self.file,
                    address,
                    header,
                })
            }
            _ => Err(Error::NotAGroup {
                path: name.to_string(),
            }),
        }
    }

    /// Open a child dataset by name.
    pub fn dataset(&self, name: &str) -> Result<Dataset<'a, R>> {
        let link = self.find_link(name)?;
        match link.target {
            LinkTarget::Hard { address } => {
                let header = ObjectHeader::parse(
                    &*self.file.reader,
                    address,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                )?;
                Ok(Dataset {
                    file: self.file,
                    address,
                    header,
                })
            }
            _ => Err(Error::NotADataset {
                path: name.to_string(),
            }),
        }
    }

    /// Read all attributes on this group.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        parse_attributes(&self.header, self.file)
    }

    /// Read all attributes on this group in creation order.
    ///
    /// Falls back to name order if creation order tracking is not available.
    pub fn attributes_by_creation_order(&self) -> Result<Vec<Attribute>> {
        parse_attributes_by_creation_order(&self.header, self.file)
    }

    /// Read all links from this group's object header.
    ///
    /// Links can be stored in two ways:
    /// 1. Directly as Link messages (0x0006) in the object header (compact storage).
    /// 2. In a fractal heap + B-tree v2, referenced by a Link Info message (0x0002).
    fn read_links(&self) -> Result<Vec<Link>> {
        let so = self.file.size_of_offsets();

        // First check for direct Link messages
        let mut links: Vec<Link> = Vec::new();
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Link {
                links.push(Link::parse(&msg.data, so)?);
            }
        }
        if !links.is_empty() {
            return Ok(links);
        }

        // Otherwise, look for Link Info message → fractal heap + B-tree v2
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::LinkInfo {
                return self.read_links_from_link_info(&msg.data);
            }
        }

        Ok(links) // empty — no links
    }

    /// Read all links, preferring creation order when available.
    fn read_links_by_creation_order(&self) -> Result<Vec<Link>> {
        let so = self.file.size_of_offsets();

        // Direct Link messages don't carry creation order — return as-is
        let mut links: Vec<Link> = Vec::new();
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Link {
                links.push(Link::parse(&msg.data, so)?);
            }
        }
        if !links.is_empty() {
            return Ok(links);
        }

        for msg in &self.header.messages {
            if msg.msg_type == MessageType::LinkInfo {
                return self.read_links_from_link_info_by_creation_order(&msg.data);
            }
        }

        Ok(links)
    }

    /// Parse a Link Info message and extract addresses.
    fn parse_link_info(&self, data: &[u8]) -> Result<(u64, u64, u64)> {
        let so = self.file.size_of_offsets();
        let o = so as usize;

        if data.len() < 2 {
            return Err(Error::InvalidObjectHeader {
                msg: "link info message too short".into(),
            });
        }

        let _version = data[0];
        let flags = data[1];
        let mut pos = 2;

        // Optional: max creation order (8 bytes, if flags bit 0 set)
        if (flags & 0x01) != 0 {
            pos += 8;
        }

        // Fractal heap address
        if pos + o > data.len() {
            return Err(Error::InvalidObjectHeader {
                msg: "link info: truncated at fractal heap address".into(),
            });
        }
        let fheap_addr = read_offset_from_slice(data, pos, so);
        pos += o;

        // B-tree v2 address (name index)
        if pos + o > data.len() {
            return Err(Error::InvalidObjectHeader {
                msg: "link info: truncated at B-tree address".into(),
            });
        }
        let bt2_name_addr = read_offset_from_slice(data, pos, so);
        pos += o;

        // Optional: B-tree v2 address (creation order index, if flags bit 1 set)
        let bt2_corder_addr = if (flags & 0x02) != 0 && pos + o <= data.len() {
            read_offset_from_slice(data, pos, so)
        } else {
            u64::MAX
        };

        Ok((fheap_addr, bt2_name_addr, bt2_corder_addr))
    }

    /// Parse a Link Info message and read links from fractal heap (name order).
    fn read_links_from_link_info(&self, data: &[u8]) -> Result<Vec<Link>> {
        let (fheap_addr, bt2_addr, _) = self.parse_link_info(data)?;
        self.read_links_from_btree(fheap_addr, bt2_addr, false)
    }

    /// Parse a Link Info message and read links in creation order.
    fn read_links_from_link_info_by_creation_order(&self, data: &[u8]) -> Result<Vec<Link>> {
        let (fheap_addr, bt2_name_addr, bt2_corder_addr) = self.parse_link_info(data)?;
        if bt2_corder_addr != u64::MAX {
            self.read_links_from_btree(fheap_addr, bt2_corder_addr, true)
        } else {
            // Fall back to name order if creation order index not available
            self.read_links_from_btree(fheap_addr, bt2_name_addr, false)
        }
    }

    /// Read links from a fractal heap using the given B-tree v2 index.
    fn read_links_from_btree(
        &self,
        fheap_addr: u64,
        bt2_addr: u64,
        by_creation_order: bool,
    ) -> Result<Vec<Link>> {
        let so = self.file.size_of_offsets();
        let sl = self.file.size_of_lengths();

        if fheap_addr == u64::MAX || bt2_addr == u64::MAX {
            return Ok(Vec::new());
        }

        let fheap = FractalHeapHeader::parse(&*self.file.reader, fheap_addr, so, sl)?;
        let bt2 = BTree2Header::parse(&*self.file.reader, bt2_addr, so, sl)?;

        let mut links = Vec::new();
        let heap_id_len = fheap.heap_id_length as usize;

        btree2::iterate_records(&*self.file.reader, &bt2, so, |record| {
            let heap_id = if by_creation_order {
                // Type 6 record: creation_order (8) + heap_id
                btree2::parse_link_creation_order_record(&record.data, heap_id_len)
                    .map(|(_order, id)| id)
            } else {
                // Type 5 record: hash (4) + heap_id
                btree2::parse_link_name_record(&record.data, heap_id_len).map(|(_hash, id)| id)
            };
            if let Some(heap_id) = heap_id {
                let link_data = fractal_heap::read_managed_object(
                    &*self.file.reader,
                    &fheap,
                    &heap_id,
                    so,
                    sl,
                )?;
                let link = Link::parse(&link_data, so)?;
                links.push(link);
            }
            Ok(())
        })?;

        Ok(links)
    }
}
