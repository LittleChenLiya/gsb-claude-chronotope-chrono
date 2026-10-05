// This is a part of Chrono.
// See README.md and LICENSE.txt for details.

//! Time-interval coverage segmentation.
//!
//! Given a batch of **timezone-aware half-open intervals** `[start, end)`, each tagged with a
//! source identifier, this module slices the timeline into segments and reports the *full set of
//! active sources* for every segment -- not merely the union of the intervals.
//!
//! The algorithm is a sweep line in UTC time:
//!
//! * Endpoints that represent the same UTC instant are grouped and processed together, even when
//!   the inputs use different UTC offsets.
//! * A source that contributes several overlapping intervals is reference counted. An interval
//!   ending therefore only removes the source when its *last* overlapping interval closes;
//!   ending one interval must never delete an owner that is still active through another one.
//! * Because the active set after every boundary is recomputed from the post-event set, two
//!   neighboring segments whose source identities differ are always emitted separately and are
//!   never blindly merged. Adjacent segments are only joined when their complete source sets are
//!   equal.
//! * Empty intervals (`start == end`) and reversed intervals (`end < start`) are diagnosed
//!   separately. A reversed interval rejects the whole batch; empty intervals are skipped while
//!   still being reported.
//! * The input slice is never modified, and every non-empty input interval is retained verbatim
//!   on the returned [`Coverage`].
//!
//! # Example
//!
//! ```
//! use chrono::coverage::{segment, SourcedInterval};
//!
//! // Two windows of owner A overlap in UTC, plus one window of owner B. The boundary at
//! // 10:00:00Z is expressed with different offsets but must group at the same instant.
//! let dt = |s: &str| chrono::DateTime::parse_from_rfc3339(s).unwrap();
//!
//! let intervals = [
//!     // A: [09:00Z, 12:00Z).
//!     SourcedInterval::new(dt("2026-10-05T09:00:00Z"), dt("2026-10-05T12:00:00Z"), "A"),
//!     // A: [10:00Z, 11:00Z), expressed at +09:00 (19:00+09 == 10:00Z).
//!     SourcedInterval::new(
//!         dt("2026-10-05T19:00:00+09:00"),
//!         dt("2026-10-05T20:00:00+09:00"),
//!         "A",
//!     ),
//!     // B: [05:00Z, 10:00Z), expressed at -05:00 (05:00-05 == 10:00Z).
//!     SourcedInterval::new(
//!         dt("2026-10-05T00:00:00-05:00"),
//!         dt("2026-10-05T05:00:00-05:00"),
//!         "B",
//!     ),
//! ];
//!
//! let coverage = segment(&intervals).unwrap();
//!
//! // A is reference counted: after the inner A window ends at 11:00Z the outer window keeps A
//! // active, so [10,11) and [11,12) form one coalesced A-only segment.
//! assert_eq!(coverage.segments.len(), 3);
//! assert_eq!(coverage.segments[0].sources, ["B"].into_iter().collect());
//! assert_eq!(coverage.segments[1].sources, ["A", "B"].into_iter().collect());
//! assert_eq!(coverage.segments[2].sources, ["A"].into_iter().collect());
//! assert_eq!(coverage.segments[2].start, dt("2026-10-05T10:00:00Z"));
//! assert_eq!(coverage.segments[2].end, dt("2026-10-05T12:00:00Z"));
//! // All three original windows are preserved, including both windows of A.
//! assert_eq!(coverage.intervals.len(), 3);
//! ```
//!
//! Run the bundled demonstration with:
//!
//! ```sh
//! cargo run --example coverage_windows
//! ```

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

use crate::{DateTime, FixedOffset};

/// A half-open, timezone-aware interval `[start, end)` tagged with a source identifier.
///
/// The endpoints may carry different UTC offsets; all comparisons are performed after converting
/// them to the same UTC instant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourcedInterval<T> {
    /// Inclusive start of the window.
    pub start: DateTime<FixedOffset>,
    /// Exclusive end of the window.
    pub end: DateTime<FixedOffset>,
    /// Identifier of the source owning this window.
    pub owner: T,
}

impl<T> SourcedInterval<T> {
    /// Creates a new sourced half-open interval `[start, end)`.
    pub fn new(start: DateTime<FixedOffset>, end: DateTime<FixedOffset>, owner: T) -> Self {
        Self { start, end, owner }
    }
}

/// The kind of problem detected while validating an input interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProblemKind {
    /// `start == end`: the window covers no time.
    Empty,
    /// `end < start`: the window runs backwards. A reversed window rejects the whole batch.
    Reversed,
}

/// A problem detected for a single input interval, located by its index in the input batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Problem {
    /// Index of the offending interval in the slice passed to [`segment`].
    pub index: usize,
    /// Whether the interval was empty or reversed.
    pub kind: ProblemKind,
}

/// Error returned when at least one interval is reversed (`end < start`).
///
/// Reversed intervals make the batch ill-posed, so no segmentation is produced and the whole
/// batch is rejected. Every reversed interval is listed, keeping reversed inputs distinct from
/// the empty-interval diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidBatch {
    /// All reversed intervals found in the batch.
    pub reversed: Vec<Problem>,
    /// All empty intervals found while scanning the (rejected) batch.
    pub empty: Vec<Problem>,
}

impl fmt::Display for InvalidBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "batch rejected: {} reversed interval(s), {} empty interval(s)",
            self.reversed.len(),
            self.empty.len()
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InvalidBatch {}

/// One maximal time span over which the set of active sources is constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment<T> {
    /// Inclusive start of the segment (a UTC instant).
    pub start: DateTime<FixedOffset>,
    /// Exclusive end of the segment (a UTC instant, always later than `start`).
    pub end: DateTime<FixedOffset>,
    /// The complete set of sources with at least one covering window over `[start, end)`.
    pub sources: BTreeSet<T>,
}

impl<T> Segment<T> {
    /// Returns `true` if `owner` is active for the whole segment.
    pub fn contains(&self, owner: &T) -> bool
    where
        T: Ord,
    {
        self.sources.contains(owner)
    }

    /// Number of distinct active sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

/// The result of segmenting a batch of sourced intervals.
///
/// Constructed with [`segment`]. The original input is preserved in two forms: [`Coverage::sources`]
/// lists every distinct owner, while [`Coverage::intervals`] keeps every non-empty input window
/// verbatim (a source may appear multiple times).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coverage<T> {
    /// Segments of constant active-source set, ordered chronologically and non-overlapping.
    /// Adjacent segments have different source sets (equal neighbors are coalesced).
    pub segments: Vec<Segment<T>>,
    /// Every distinct source that contributes at least one non-empty interval.
    pub sources: BTreeSet<T>,
    /// Every non-empty input interval, in input order, retained verbatim.
    pub intervals: Vec<SourcedInterval<T>>,
    /// Locations of empty (`start == end`) input intervals, which were skipped.
    pub empty: Vec<Problem>,
}

enum EventKind {
    Start,
    End,
}

struct Event {
    at: DateTime<FixedOffset>,
    kind: EventKind,
    interval_index: usize,
}

/// Segments a batch of cross-timezone, sourced half-open intervals.
///
/// Returns [`InvalidBatch`] if any interval is reversed (`end < start`), in which case the whole
/// batch is rejected and `Ok` is never produced. Empty intervals (`start == end`) do not reject
/// the batch: they are reported in [`Coverage::empty`] and skipped.
///
/// The input slice is never modified.
///
/// # Reference counting
///
/// The same owner may supply multiple overlapping intervals. Internally an owner carries a
/// reference count equal to the number of its intervals currently covering the sweep point, so an
/// interval ending only removes the owner once the count reaches zero.
///
/// # Endpoint grouping
///
/// All endpoints sharing one UTC instant are processed together before any segment ending or
/// starting at that instant is finalized. Within such a group, end events are applied before
/// start events to give the half-open `[start, end)` semantics.
pub fn segment<T>(intervals: &[SourcedInterval<T>]) -> Result<Coverage<T>, InvalidBatch>
where
    T: Clone + Ord,
{
    let mut reversed = Vec::new();
    let mut empty = Vec::new();

    // Separate diagnostics for empty versus reversed intervals; reject the whole batch if any
    // reversed interval is present.
    for (index, interval) in intervals.iter().enumerate() {
        match interval.end.cmp(&interval.start) {
            core::cmp::Ordering::Less => {
                reversed.push(Problem { index, kind: ProblemKind::Reversed });
            }
            core::cmp::Ordering::Equal => {
                empty.push(Problem { index, kind: ProblemKind::Empty });
            }
            core::cmp::Ordering::Greater => {}
        }
    }

    if !reversed.is_empty() {
        return Err(InvalidBatch { reversed, empty });
    }

    let mut events: Vec<Event> = Vec::with_capacity(intervals.len() * 2);
    let mut retained: Vec<SourcedInterval<T>> = Vec::with_capacity(intervals.len());
    let mut sources: BTreeSet<T> = BTreeSet::new();

    for (index, interval) in intervals.iter().enumerate() {
        if interval.start == interval.end {
            continue;
        }
        events.push(Event { at: interval.start, kind: EventKind::Start, interval_index: index });
        events.push(Event { at: interval.end, kind: EventKind::End, interval_index: index });
        sources.insert(interval.owner.clone());
        retained.push(interval.clone());
    }

    // Order by UTC instant. Endpoints at the same UTC instant sort together regardless of their
    // original offset; within an instant, ends are applied before starts (half-open semantics).
    events.sort_by(|a, b| match a.at.cmp(&b.at) {
        core::cmp::Ordering::Equal => match (&a.kind, &b.kind) {
            (EventKind::End, EventKind::Start) => core::cmp::Ordering::Less,
            (EventKind::Start, EventKind::End) => core::cmp::Ordering::Greater,
            _ => a.interval_index.cmp(&b.interval_index),
        },
        other => other,
    });

    // Number of intervals of each owner covering the current sweep point.
    let mut refcounts: BTreeMap<T, usize> = BTreeMap::new();
    let mut segments: Vec<Segment<T>> = Vec::new();

    let mut cursor = events.first().map(|event| event.at);
    let mut next_event = 0usize;

    while next_event < events.len() {
        let boundary = events[next_event].at;

        // Close the segment that was open up to this boundary. The first boundary only opens a
        // span, so there is nothing to close there.
        if let Some(open_start) = cursor {
            if boundary > open_start {
                let active: BTreeSet<T> = refcounts
                    .iter()
                    .filter(|(_, &count)| count > 0)
                    .map(|(owner, _)| owner.clone())
                    .collect();
                if !active.is_empty() {
                    push_segment(&mut segments, open_start, boundary, active);
                }
            }
        }

        // Apply every event at this exact UTC instant together.
        while next_event < events.len() && events[next_event].at == boundary {
            let event = &events[next_event];
            let owner = &intervals[event.interval_index].owner;
            match event.kind {
                EventKind::Start => {
                    *refcounts.entry(owner.clone()).or_insert(0) += 1;
                }
                EventKind::End => {
                    let count =
                        refcounts.get_mut(owner).expect("end event without matching start event");
                    *count -= 1;
                    // Only drop the owner when its last overlapping window closes.
                    if *count == 0 {
                        refcounts.remove(owner);
                    }
                }
            }
            next_event += 1;
        }

        cursor = Some(boundary);
    }

    Ok(Coverage { segments, sources, intervals: retained, empty })
}

fn push_segment<T>(
    segments: &mut Vec<Segment<T>>,
    start: DateTime<FixedOffset>,
    end: DateTime<FixedOffset>,
    sources: BTreeSet<T>,
) where
    T: Ord,
{
    // Do not blindly merge neighbors: join the previous segment only if its source identity is
    // exactly the same.
    if let Some(previous) = segments.last_mut() {
        if previous.end == start && previous.sources == sources {
            previous.end = end;
            return;
        }
    }
    segments.push(Segment { start, end, sources });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeZone;
    use alloc::vec;

    fn utc(hour: u32) -> DateTime<FixedOffset> {
        crate::Utc.with_ymd_and_hms(2026, 10, 5, hour, 0, 0).unwrap().fixed_offset()
    }

    fn window(
        start: DateTime<FixedOffset>,
        end: DateTime<FixedOffset>,
        owner: &str,
    ) -> SourcedInterval<&str> {
        SourcedInterval::new(start, end, owner)
    }

    #[test]
    fn two_overlapping_windows_then_b_segmentation() {
        // A owns two overlapping windows; B joins and leaves independently.
        let intervals = [
            window(utc(9), utc(12), "A"),
            window(utc(10), utc(11), "A"),
            window(utc(5), utc(10), "B"),
        ];

        let coverage = segment(&intervals).unwrap();

        // A is reference counted, so after the inner A window closes at 11:00 the outer window
        // keeps the source set identical to [10,11): those two neighbors coalesce.
        let expected: Vec<(u32, u32, &[&str])> =
            vec![(5, 9, &["B"]), (9, 10, &["A", "B"]), (10, 12, &["A"])];
        assert_eq!(coverage.segments.len(), expected.len());
        for (segment, (start, end, owners)) in coverage.segments.iter().zip(expected) {
            assert_eq!(segment.start, utc(start));
            assert_eq!(segment.end, utc(end));
            let expected_sources: BTreeSet<&str> = owners.iter().copied().collect();
            assert_eq!(segment.sources, expected_sources);
        }

        // The inner and outer windows of A are both retained.
        assert_eq!(coverage.intervals.len(), 3);
        assert_eq!(coverage.sources, ["A", "B"].into_iter().collect());
        assert!(coverage.empty.is_empty());
    }

    #[test]
    fn closing_inner_window_while_other_source_lingers_splits_segments() {
        // Same layout but C covers the inner A window: the set changes at 11:00, proving that
        // segments are split by source identity, not blindly by endpoints.
        let intervals = [
            window(utc(9), utc(12), "A"),
            window(utc(10), utc(11), "A"),
            window(utc(5), utc(10), "B"),
            window(utc(10), utc(11), "C"),
        ];

        let coverage = segment(&intervals).unwrap();
        let expected: Vec<(u32, u32, &[&str])> =
            vec![(5, 9, &["B"]), (9, 10, &["A", "B"]), (10, 11, &["A", "C"]), (11, 12, &["A"])];
        assert_eq!(coverage.segments.len(), expected.len());
        for (segment, (start, end, owners)) in coverage.segments.iter().zip(expected) {
            assert_eq!(segment.start, utc(start));
            assert_eq!(segment.end, utc(end));
            assert_eq!(segment.sources, owners.iter().copied().collect());
        }
    }

    #[test]
    fn same_instant_different_offsets_groups_endpoints() {
        // 19:00+09 == 10:00Z and 05:00-05 == 10:00Z: the shared boundary must be one instant
        // even though the local wall-clock values and offsets differ.
        let plus9 = FixedOffset::east_opt(9 * 3600).unwrap();
        let minus5 = FixedOffset::west_opt(5 * 3600).unwrap();
        let a_start =
            crate::Utc.with_ymd_and_hms(2026, 10, 5, 9, 0, 0).unwrap().with_timezone(&plus9);
        let a_end =
            crate::Utc.with_ymd_and_hms(2026, 10, 5, 10, 0, 0).unwrap().with_timezone(&plus9);
        let b_start =
            crate::Utc.with_ymd_and_hms(2026, 10, 5, 5, 0, 0).unwrap().with_timezone(&minus5);
        let b_end = crate::Utc.with_ymd_and_hms(2026, 10, 5, 10, 0, 0).unwrap().fixed_offset();
        let intervals = [window(a_start, a_end, "A"), window(b_start, b_end, "B")];

        let coverage = segment(&intervals).unwrap();
        assert_eq!(coverage.segments.len(), 2);
        assert_eq!(coverage.segments[0].start, utc(5));
        assert_eq!(coverage.segments[0].end, utc(9));
        assert_eq!(coverage.segments[0].sources, ["B"].into_iter().collect());
        assert_eq!(coverage.segments[1].start, utc(9));
        assert_eq!(coverage.segments[1].end, utc(10));
        assert_eq!(coverage.segments[1].sources, ["A", "B"].into_iter().collect());
    }

    #[test]
    fn reversed_interval_rejects_whole_batch() {
        let intervals = [
            window(utc(9), utc(12), "A"),
            window(utc(11), utc(10), "B"), // reversed
        ];

        let error = segment(&intervals).expect_err("batch must be rejected");
        assert_eq!(error.reversed, vec![Problem { index: 1, kind: ProblemKind::Reversed }]);
        assert!(error.empty.is_empty());
    }

    #[test]
    fn reversed_and_empty_are_diagnosed_separately() {
        let intervals = [
            window(utc(8), utc(8), "E"),
            window(utc(11), utc(9), "R"), // reversed
        ];

        let error = segment(&intervals).unwrap_err();
        assert_eq!(error.reversed.len(), 1);
        assert_eq!(error.reversed[0].index, 1);
        assert_eq!(error.empty.len(), 1);
        assert_eq!(error.empty[0].index, 0);
    }

    #[test]
    fn empty_intervals_are_skipped_and_reported() {
        let intervals = [window(utc(9), utc(12), "A"), window(utc(10), utc(10), "empty")];

        let coverage = segment(&intervals).unwrap();
        assert_eq!(coverage.empty, vec![Problem { index: 1, kind: ProblemKind::Empty }]);
        assert_eq!(coverage.intervals.len(), 1);
        assert_eq!(coverage.sources, ["A"].into_iter().collect());
        assert_eq!(coverage.segments.len(), 1);
        assert_eq!(coverage.segments[0].sources, ["A"].into_iter().collect());
    }

    #[test]
    fn refcount_keeps_owner_until_last_window_closes() {
        // Two overlapping windows of A: ending the inner one must not remove A.
        let intervals = [window(utc(9), utc(12), "A"), window(utc(10), utc(11), "A")];

        let coverage = segment(&intervals).unwrap();
        // Sources identical on [9,12), so the two neighbors coalesce into one A-only segment.
        assert_eq!(coverage.segments.len(), 1);
        assert_eq!(coverage.segments[0].start, utc(9));
        assert_eq!(coverage.segments[0].end, utc(12));
        assert_eq!(coverage.segments[0].sources, ["A"].into_iter().collect());
    }

    #[test]
    fn adjacent_different_identities_are_never_merged() {
        // A ends exactly where B begins: two segments must survive, even though they touch.
        let intervals = [window(utc(9), utc(10), "A"), window(utc(10), utc(11), "B")];

        let coverage = segment(&intervals).unwrap();
        assert_eq!(coverage.segments.len(), 2);
        assert_eq!(coverage.segments[0].sources, ["A"].into_iter().collect());
        assert_eq!(coverage.segments[1].sources, ["B"].into_iter().collect());
        assert_eq!(coverage.segments[0].end, coverage.segments[1].start);
    }

    #[test]
    fn input_slice_is_not_mutated() {
        let intervals = [
            SourcedInterval::new(utc(9), utc(12), "A"),
            SourcedInterval::new(utc(8), utc(8), "empty"),
        ];
        let snapshot = intervals.clone();

        let _ = segment(&intervals).unwrap();
        assert_eq!(intervals, snapshot);
    }
}
