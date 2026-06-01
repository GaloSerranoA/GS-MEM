use time::OffsetDateTime;

use crate::types::{Link, Slug};

pub fn open_edge(
    from_slug: Slug,
    to_slug: Slug,
    edge_type: impl Into<String>,
    valid_at: OffsetDateTime,
) -> Link {
    Link {
        id: None,
        from_slug,
        to_slug,
        edge_type: edge_type.into(),
        valid_at,
        invalid_at: None,
    }
}

pub fn close_edge(link: &mut Link, invalid_at: OffsetDateTime) {
    link.invalid_at = Some(invalid_at);
}

pub fn query_at(links: &[Link], t: OffsetDateTime) -> Vec<&Link> {
    links
        .iter()
        .filter(|link| link.valid_at <= t && link.invalid_at.map(|ts| ts > t).unwrap_or(true))
        .collect()
}
