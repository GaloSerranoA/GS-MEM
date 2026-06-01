use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchStrategy {
    FullText,
    Vector,
    TagIntersection,
    BacklinkHop { max_depth: u8 },
    Hybrid { rrf_k: u32 },
}
