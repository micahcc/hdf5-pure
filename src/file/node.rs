use crate::file::dataset::Dataset;
use crate::file::group::Group;
use crate::io::ReadAt;

/// A node in the HDF5 hierarchy — either a group or a dataset.
pub enum Node<'a, R: ReadAt + ?Sized> {
    Group(Group<'a, R>),
    Dataset(Dataset<'a, R>),
}
