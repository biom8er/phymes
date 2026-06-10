use std::fmt::Display;

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::DataFormat;

/// Data compression and decompression encodings
#[derive(Clone, Debug, PartialEq, Eq, Serialize, ValueEnum, Deserialize, Default, Hash)]
pub enum DataEncoding {
    /// Deflate
    #[value(name = "Deflate")]
    Deflate,
    /// Zlib
    #[value(name = "Zlib")]
    Zlib,
    /// Gz
    #[value(name = "Gz")]
    Gz,
    #[default]
    #[value(name = "None")]
    None,
}

impl DataEncoding {
    /// Convert from a filename prefix
    pub fn from_extension(extension: &str) -> Result<Self> {
        let format = match extension {
            "deflate" => DataEncoding::Deflate,
            "" => DataEncoding::Deflate,
            "zz" => DataEncoding::Zlib,
            "zlib" => DataEncoding::Zlib,
            "gz" => DataEncoding::Gz,
            _ => {
                return Err(anyhow!(
                    "File extension {extension} was not recognized. Supported extensions are .deflate, .zz, .zlib, and .gz"
                ));
            }
        };
        Ok(format)
    }

    /// The file prefix for the format
    pub fn to_extension(&self) -> &str {
        match self {
            Self::Deflate => "",
            Self::Zlib => "zz",
            Self::Gz => "gz",
            Self::None => "",
        }
    }
}

impl Display for DataEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Deflate => write!(f, "Deflate"),
            Self::Zlib => write!(f, "Zlib"),
            Self::Gz => write!(f, "Gz"),
            Self::None => write!(f, "None"),
        }
    }
}

/// Make the filename based on the [DataFormat] and the [DataEncoding]
///
/// # Notes
/// - filename: "foo" of "/bar/foo.rs.gz"
/// - prefix: "rs" of "/bar/foo.rs.gz"
/// - extension: "gz" of "/bar/foo.rs.gz"
pub fn make_filename(filename: &str, format: &DataFormat, encoding: &DataEncoding) -> String {
    let mut filename_vec = vec![filename];
    if format != &DataFormat::None {
        filename_vec.push(format.to_prefix());
    }
    if encoding != &DataEncoding::None {
        filename_vec.push(encoding.to_extension());
    }
    filename_vec.join(".")
}

/// Make the extension based on the [DataFormat] and the [DataEncoding]
///
/// # Notes
/// - we treat everything after the "." as a part of the extension including the prefix and extension
/// - prefix: "rs" of "/bar/foo.rs.gz"
/// - extension: "gz" of "/bar/foo.rs.gz"
pub fn make_extension(format: &DataFormat, encoding: &DataEncoding) -> String {
    let mut filename_vec = Vec::new();
    if format != &DataFormat::None {
        filename_vec.push(format.to_prefix());
    }
    if encoding != &DataEncoding::None {
        filename_vec.push(encoding.to_extension());
    }
    filename_vec.join(".")
}
