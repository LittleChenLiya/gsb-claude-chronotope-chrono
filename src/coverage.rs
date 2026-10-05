//! Coverage segmentation for half-open time intervals.
//!
//! This module turns a collection of half-open intervals `[start, end)`, each
//! tagged with a source identifier, into a *segmentation* of the timeline: the
//! timeline is split at every interval endpoint, and each segment is annotated
//! with the complete set of sources that are active throughout it.
//!
//! # Semantics
//!
//! * Intervals are half-open: `[start, end)` includes the start instant and
//!   excludes the end instant.
//! * Endpoints are normalized to UTC before comparison, so intervals expressed
//!   in different timezones combine consistently: two endpoints at the same
//!   instant are processed together regardless of their offset.
//! * A source may declare several intervals. Reference counting is used so that
//!   ending one interval does not remove the source while another of its
//!   intervals still covers the segment.
//! * Empty intervals (`start == end`) and reverse intervals (`start > end`) are
//!   diagnosed separately; a reverse interval rejects the whole batch.
//! * Adjacent segments are merged only when they share the same source set.
//!   Segments with different source identities are never merged.
//!
//! # Example
//!
//! ```
//! use chrono::{SourceInterval, TimeZone, Utc};
//!
//! let t = |h| Utc.with_ymd_and_hms(2026, 1, 1, h, 0, 0).unwrap();
//! let intervals = vec![
//!     SourceInterval::new("A", t(0), t(10)),
//!     SourceInterval::new("A", t(5), t(15)),
//!     SourceInterval::new("B", t(12), t(20)),
//! ];
//!
//! let segments = chrono::coverage(&intervals).unwrap();
//! assert_eq!(segments.len(), 3);
//!
//! assert_eq!(segments[0].start, t(0));
//! assert_eq!(segments[0].end, t(12));
//! assert_eq!(segments[0].sources.iter().map(|s| s.id).collect::<Vec<_>>(), vec!["A"]);
//!
//! assert_eq!(segments[1].start, t(12));
//! assert_eq!(segments[1].end, t(15));
//! assert_eq!(segments[1].sources.iter().map(|s| s.id).collect::<Vec<_>>(), vec!["A", "B"]);
//!
//! assert_eq!(segments[2].start, t(15));
//! assert_eq!(segments[2].end, t(20));
//! assert_eq!(segments[2].sources.iter().map(|s| s.id).collect::<Vec<_>>(), vec!["B"]);
//! ```

use crate::{DateTime, TimeZone, Utc};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

/// A half-open time interval `[start, end)` tagged with a source identifier.
///
/// Both endpoints are stored in their original [`TimeZone`] and are only
/// normalized to UTC when compared, so intervals expressed in different
/// timezones combine consistently.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceInterval<Id, Tz: TimeZone> {
    /// Identifier of the source this interval belongs to.
    pub id: Id,
    /// Start instant (inclusive), kept in its original timezone.
    pub start: DateTime<Tz>,
    /// End instant (exclusive), kept in its original timezone.
    pub end: DateTime<Tz>,
}

impl<Id, Tz: TimeZone> SourceInterval<Id, Tz> {
    /// Creates a new source interval.
    pub fn new(id: Id, start: DateTime<Tz>, end: DateTime<Tz>) -> Self {
        SourceInterval { id, start, end }
    }
}

/// A segment of the timeline in which a fixed set of sources is active.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment<Id, Tz: TimeZone> {
    /// Start instant of the segment (inclusive), in UTC.
    pub start: DateTime<Utc>,
    /// End instant of the segment (exclusive), in UTC.
    pub end: DateTime<Utc>,
    /// Sources active throughout the segment, sorted by identifier.
    pub sources: Vec<ActiveSource<Id, Tz>>,
}

/// A source active within a [`Segment`], together with the original interval(s)
/// it was declared with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveSource<Id, Tz: TimeZone> {
    /// Source identifier.
    pub id: Id,
    /// Original intervals of this source that overlap the segment, sorted by
    /// start instant.
    pub intervals: Vec<SourceInterval<Id, Tz>>,
}

/// An error that prevents coverage segmentation.
///
/// Empty intervals (`start == end`) and reverse intervals (`start > end`) use
/// distinct variants so they can be diagnosed separately.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoverageError {
    /// An interval has the same start and end instant `[t, t)`.
    EmptyInterval {
        /// Index of the offending interval in the input slice.
        index: usize,
    },
    /// An interval starts after it ends `[t, s)` with `t > s`.
    ReverseInterval {
        /// Index of the offending interval in the input slice.
        index: usize,
    },
}

impl fmt::Display for CoverageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CoverageError::EmptyInterval { index } => {
                write!(f, "empty interval at index {index}: start and end are the same instant")
            }
            CoverageError::ReverseInterval { index } => {
                write!(f, "reverse interval at index {index}: start is after end")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CoverageError {}

#[cfg(all(not(feature = "std"), feature = "core-error"))]
impl core::error::Error for CoverageError {}

/// Computes the coverage segmentation of a collection of half-open intervals.
///
/// The timeline is split at every interval endpoint, and each segment is
/// annotated with the complete set of sources active throughout it. Adjacent
/// segments are merged only when they share the same source set; segments with
/// different source identities are never merged.
///
/// The original input is not modified.
///
/// # Errors
///
/// Returns [`CoverageError::EmptyInterval`] if any interval has `start == end`,
/// or [`CoverageError::ReverseInterval`] if any interval has `start > end`. A
/// reverse interval rejects the whole batch.
pub fn coverage<Id, Tz>(
    intervals: &[SourceInterval<Id, Tz>],
) -> Result<Vec<Segment<Id, Tz>>, CoverageError>
where
    Id: Ord + Clone,
    Tz: TimeZone,
{
    // Normalize endpoints to UTC and validate each interval. The whole batch is
    // rejected on the first invalid interval.
    let normalized: Vec<(DateTime<Utc>, DateTime<Utc>)> = intervals
        .iter()
        .enumerate()
        .map(|(index, iv)| {
            let start = iv.start.with_timezone(&Utc);
            let end = iv.end.with_timezone(&Utc);
            if start == end {
                Err(CoverageError::EmptyInterval { index })
            } else if start > end {
                Err(CoverageError::ReverseInterval { index })
            } else {
                Ok((start, end))
            }
        })
        .collect::<Result<_, _>>()?;

    // Build the sweep events. A start adds one to the reference count of the
    // owner; an end subtracts one. Events at the same instant are processed
    // together so that an interval ending at `t` and another starting at `t` do
    // not create a spurious empty segment.
    let mut events: Vec<(DateTime<Utc>, i64, usize)> = Vec::with_capacity(2 * intervals.len());
    for (i, (start, end)) in normalized.iter().enumerate() {
        events.push((*start, 1, i));
        events.push((*end, -1, i));
    }
    events.sort_by_key(|a| a.0);

    // Sweep, grouping all events at the same instant.
    let mut counts: BTreeMap<Id, i64> = BTreeMap::new();
    let mut active: BTreeSet<Id> = BTreeSet::new();
    // `(instant, set active for [instant, next instant))`
    let mut boundaries: Vec<(DateTime<Utc>, BTreeSet<Id>)> = Vec::new();

    let mut i = 0;
    while i < events.len() {
        let t = events[i].0;
        while i < events.len() && events[i].0 == t {
            let (_, delta, idx) = events[i];
            let id = &intervals[idx].id;
            let count = counts.entry(id.clone()).or_insert(0);
            *count += delta;
            if *count == 0 {
                active.remove(id);
            } else {
                active.insert(id.clone());
            }
            i += 1;
        }
        boundaries.push((t, active.clone()));
    }

    // Build maximal segments: merge adjacent boundaries that share the same
    // active set, but never merge segments with different source identities.
    let mut segments: Vec<Segment<Id, Tz>> = Vec::new();
    let mut i = 0;
    while i < boundaries.len() {
        let (start, set) = &boundaries[i];
        if set.is_empty() {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < boundaries.len() && &boundaries[j].1 == set {
            j += 1;
        }
        // The segment spans [start, boundaries[j].0). The last boundary has an
        // empty set (all intervals have ended), so `j` always lands on a
        // different set.
        let end = boundaries[j].0;

        let mut sources: Vec<ActiveSource<Id, Tz>> = Vec::new();
        for id in set.iter() {
            let mut overlapping: Vec<SourceInterval<Id, Tz>> = Vec::new();
            for (k, iv) in intervals.iter().enumerate() {
                if &iv.id == id {
                    let (s, e) = normalized[k];
                    // Half-open overlap: [s, e) overlaps [start, end).
                    if s < end && e > *start {
                        overlapping.push(iv.clone());
                    }
                }
            }
            overlapping.sort_by_key(|iv| iv.start.clone());
            sources.push(ActiveSource { id: id.clone(), intervals: overlapping });
        }

        segments.push(Segment { start: *start, end, sources });
        i = j;
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedOffset, TimeZone, Utc};
    use alloc::vec;

    fn dt(h: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, h, 0, 0).unwrap()
    }

    fn ids<'a, Tz: TimeZone>(seg: &'a Segment<&'a str, Tz>) -> Vec<&'a str> {
        seg.sources.iter().map(|s| s.id).collect()
    }

    #[test]
    fn overlapping_intervals_with_refcount() {
        let intervals = vec![
            SourceInterval::new("A", dt(0), dt(10)),
            SourceInterval::new("A", dt(5), dt(15)),
            SourceInterval::new("B", dt(12), dt(20)),
        ];
        let segments = coverage(&intervals).unwrap();
        assert_eq!(segments.len(), 3);

        // [0, 12): {A} — A is not dropped when its first interval ends at 10.
        assert_eq!(segments[0].start, dt(0));
        assert_eq!(segments[0].end, dt(12));
        assert_eq!(ids(&segments[0]), vec!["A"]);
        // Both of A's original intervals are retained.
        assert_eq!(segments[0].sources[0].intervals.len(), 2);

        // [12, 15): {A, B}
        assert_eq!(segments[1].start, dt(12));
        assert_eq!(segments[1].end, dt(15));
        assert_eq!(ids(&segments[1]), vec!["A", "B"]);

        // [15, 20): {B}
        assert_eq!(segments[2].start, dt(15));
        assert_eq!(segments[2].end, dt(20));
        assert_eq!(ids(&segments[2]), vec!["B"]);
    }

    #[test]
    fn reverse_interval_rejects_whole_batch() {
        let intervals =
            vec![SourceInterval::new("A", dt(0), dt(10)), SourceInterval::new("B", dt(20), dt(10))];
        let err = coverage(&intervals).unwrap_err();
        assert_eq!(err, CoverageError::ReverseInterval { index: 1 });
    }

    #[test]
    fn empty_interval_diagnosed_separately() {
        let intervals =
            vec![SourceInterval::new("A", dt(0), dt(10)), SourceInterval::new("B", dt(5), dt(5))];
        let err = coverage(&intervals).unwrap_err();
        assert_eq!(err, CoverageError::EmptyInterval { index: 1 });
    }

    #[test]
    fn original_input_unchanged() {
        let intervals = vec![SourceInterval::new("A", dt(0), dt(10))];
        let before = intervals.clone();
        let _ = coverage(&intervals).unwrap();
        assert_eq!(intervals, before);
    }

    #[test]
    fn cross_timezone_same_instant() {
        let p1 = FixedOffset::east_opt(3600).unwrap();
        let m5 = FixedOffset::west_opt(5 * 3600).unwrap();
        // A: [09:00+01:00, 12:00+01:00) = [08:00Z, 11:00Z)
        // B: [03:00-05:00, 07:00-05:00) = [08:00Z, 12:00Z)
        // Both start at the same instant 08:00Z despite different offsets.
        let intervals = vec![
            SourceInterval::new("A", dt(8).with_timezone(&p1), dt(11).with_timezone(&p1)),
            SourceInterval::new("B", dt(8).with_timezone(&m5), dt(12).with_timezone(&m5)),
        ];
        let segments = coverage(&intervals).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start, dt(8));
        assert_eq!(segments[0].end, dt(11));
        assert_eq!(ids(&segments[0]), vec!["A", "B"]);
        assert_eq!(segments[1].start, dt(11));
        assert_eq!(segments[1].end, dt(12));
        assert_eq!(ids(&segments[1]), vec!["B"]);
    }

    #[test]
    fn adjacent_different_identities_not_merged() {
        let intervals =
            vec![SourceInterval::new("A", dt(0), dt(10)), SourceInterval::new("B", dt(10), dt(20))];
        let segments = coverage(&intervals).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].end, dt(10));
        assert_eq!(segments[1].start, dt(10));
    }

    #[test]
    fn empty_input() {
        let intervals: Vec<SourceInterval<&str, Utc>> = vec![];
        let segments = coverage(&intervals).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn same_instant_end_and_start_of_same_owner() {
        // The first interval of A ends at 10 and its second starts at 10. The
        // owner must not be dropped at the boundary: the segment [0, 20) has
        // {A}.
        let intervals =
            vec![SourceInterval::new("A", dt(0), dt(10)), SourceInterval::new("A", dt(10), dt(20))];
        let segments = coverage(&intervals).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, dt(0));
        assert_eq!(segments[0].end, dt(20));
        assert_eq!(ids(&segments[0]), vec!["A"]);
        assert_eq!(segments[0].sources[0].intervals.len(), 2);
    }

    #[test]
    fn three_window_example() {
        // Three windows with cross-timezone endpoints, mirroring the example.
        let p1 = FixedOffset::east_opt(3600).unwrap();
        let m5 = FixedOffset::west_opt(5 * 3600).unwrap();
        let p0 = FixedOffset::east_opt(0).unwrap();
        let intervals = vec![
            // Window A: two overlapping intervals in +01:00
            SourceInterval::new("A", dt(8).with_timezone(&p1), dt(11).with_timezone(&p1)),
            SourceInterval::new("A", dt(10).with_timezone(&p1), dt(13).with_timezone(&p1)),
            // Window B: UTC
            SourceInterval::new("B", dt(12).with_timezone(&p0), dt(16).with_timezone(&p0)),
            // Window C: -05:00
            SourceInterval::new("C", dt(10).with_timezone(&m5), dt(15).with_timezone(&m5)),
        ];
        let segments = coverage(&intervals).unwrap();
        // C is active 10:00-15:00; A's two intervals overlap so the {A, C}
        // segment merges across the 11:00 boundary.
        let expected: Vec<(u32, u32, Vec<&str>)> = vec![
            (8, 10, vec!["A"]),
            (10, 12, vec!["A", "C"]),
            (12, 13, vec!["A", "B", "C"]),
            (13, 15, vec!["B", "C"]),
            (15, 16, vec!["B"]),
        ];
        assert_eq!(segments.len(), expected.len());
        for (seg, (s, e, set)) in segments.iter().zip(expected.iter()) {
            assert_eq!(seg.start, dt(*s));
            assert_eq!(seg.end, dt(*e));
            assert_eq!(ids(seg), *set);
        }
    }
}
