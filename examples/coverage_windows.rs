//! Three-window demonstration of `chrono::coverage`.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example coverage_windows
//! ```
//!
//! Windows:
//!
//! * Source A, window 1: `[09:00Z, 12:00Z)` written in UTC.
//! * Source A, window 2: `[10:00Z, 11:00Z)` written at `+09:00` (19:00+09 == 10:00Z).
//! * Source B: `[05:00Z, 10:00Z)` written at `-05:00` (05:00-05 == 10:00Z).
//!
//! The two windows of A overlap, so A is reference counted: when the inner window closes at
//! 11:00Z the outer window still keeps A active until 12:00Z. The 10:00Z boundary is reached by
//! B with offset `-05:00` and by A with offset `+09:00`; both are the same UTC instant and are
//! processed together.

use chrono::DateTime;
use chrono::coverage::{SourcedInterval, segment};

fn main() {
    let parse = DateTime::parse_from_rfc3339;

    let intervals = [
        SourcedInterval::new(
            parse("2026-10-05T09:00:00Z").unwrap(),
            parse("2026-10-05T12:00:00Z").unwrap(),
            "A",
        ),
        SourcedInterval::new(
            parse("2026-10-05T19:00:00+09:00").unwrap(),
            parse("2026-10-05T20:00:00+09:00").unwrap(),
            "A",
        ),
        SourcedInterval::new(
            parse("2026-10-05T00:00:00-05:00").unwrap(),
            parse("2026-10-05T05:00:00-05:00").unwrap(),
            "B",
        ),
    ];

    let coverage = segment(&intervals).expect("no reversed intervals in this example");

    println!("sources: {:?}", coverage.sources);
    println!("retained original windows: {}", coverage.intervals.len());
    for interval in &coverage.intervals {
        println!("  {}: [{} .. {})", interval.owner, interval.start, interval.end);
    }

    println!("segments:");
    for segment in &coverage.segments {
        let owners: Vec<&str> = segment.sources.iter().copied().collect();
        println!("  [{} .. {}): active = {:?}", segment.start, segment.end, owners);
    }

    // Expected timeline (UTC). A is reference counted: after the inner A window ends at 11:00Z
    // the outer window keeps A active, so the two A-only spans coalesce into one segment:
    //   [05:00, 09:00): {B}
    //   [09:00, 10:00): {A, B}   <- 10:00Z joins -05:00 and +09:00 endpoints
    //   [10:00, 12:00): {A}      <- inner A closes at 11:00, outer A stays until 12:00
    assert_eq!(coverage.segments.len(), 3);
    assert_eq!(coverage.intervals.len(), 3);
}
