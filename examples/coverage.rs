//! Three-window coverage segmentation example.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example coverage
//! ```
//!
//! Three windows are declared in different timezones:
//!
//! * Window `A` has two overlapping intervals in `+01:00`, exercising the
//!   reference counting of a source with several intervals.
//! * Window `B` is in UTC.
//! * Window `C` is in `-05:00`.
//!
//! The endpoints are normalized to UTC, so two endpoints at the same instant
//! (different offsets) are processed together.

use chrono::{DateTime, FixedOffset, SourceInterval};

fn rfc3339(s: &str) -> DateTime<FixedOffset> {
    DateTime::parse_from_rfc3339(s).unwrap()
}

fn main() {
    let intervals = vec![
        // Window A: two overlapping intervals in +01:00.
        SourceInterval::new(
            "A",
            rfc3339("2026-01-01T09:00:00+01:00"),
            rfc3339("2026-01-01T12:00:00+01:00"),
        ),
        SourceInterval::new(
            "A",
            rfc3339("2026-01-01T11:00:00+01:00"),
            rfc3339("2026-01-01T14:00:00+01:00"),
        ),
        // Window B: UTC.
        SourceInterval::new("B", rfc3339("2026-01-01T12:00:00Z"), rfc3339("2026-01-01T16:00:00Z")),
        // Window C: -05:00.
        SourceInterval::new(
            "C",
            rfc3339("2026-01-01T05:00:00-05:00"),
            rfc3339("2026-01-01T10:00:00-05:00"),
        ),
    ];

    let segments = chrono::coverage(&intervals).unwrap();
    for seg in &segments {
        let ids: Vec<_> = seg.sources.iter().map(|s| s.id).collect();
        println!("[{}, {}) -> {{ {} }}", seg.start, seg.end, ids.join(", "));
        for source in &seg.sources {
            for iv in &source.intervals {
                println!("    {}: [{}, {})", source.id, iv.start, iv.end);
            }
        }
    }
}
