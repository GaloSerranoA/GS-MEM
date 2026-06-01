use serde::{Deserialize, Serialize};

use crate::fingerprint::Fingerprint;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceVersion {
    pub mtime_us: i64,
    pub content_fp: Fingerprint,
    pub logic_fp: Fingerprint,
}
