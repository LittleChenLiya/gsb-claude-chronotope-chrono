// This is a part of Chrono.
// See README.md and LICENSE.txt for details.

//! Splitting a UTC half-open interval into segments with a constant local UTC offset.
//!
//! The caller provides:
//!
//! * a *time zone adapter* (anything implementing [`OffsetAt`]) that can report the UTC offset
//!   in effect at a requested instant, and
//! * a *candidate transition set*: the sorted list of UTC timestamps (whole seconds) at which the
//!   offset *may* change.
//!
//! [`split_interval`] then checks the offset immediately before and at every candidate — second
//! level resolution, no per-second scanning — and cuts the interval wherever the offset actually
//! changes. Because every candidate is checked individually, transitions that cancel out over the
//! whole interval (e.g. the offset increases and later returns to its original value) are still
//! detected, unlike a naive comparison of only the interval endpoints.
//!
//! # Contract
//!
//! * **Completeness of the candidate set is guaranteed by the caller.** Any offset transition not
//!   present in `candidates` will not be detected. Candidates at which the offset does not
//!   actually change are harmless and simply do not produce a split.
//! * Candidates must be sorted in non-decreasing order; duplicates are allowed and ignored.
//!   Unsorted input is rejected with [`SegmentError::UnsortedCandidates`].
//! * The interval is half-open: `[start, end)`. An instant exactly at a transition belongs to the
//!   *new* offset (the offset is queried *at* the transition second, and at `transition - 1` for
//!   the "before" side).
//! * Adjacent segments with the same offset are merged, and zero-length segments are never
//!   emitted. A zero-length (or empty) input interval yields an empty result.

use alloc::vec::Vec;
use core::fmt;

/// The largest accepted absolute UTC offset, in seconds (24 hours).
///
/// This mirrors the range accepted by [`FixedOffset`](crate::FixedOffset).
const MAX_OFFSET_SECS: i64 = 86_400;

/// A time zone adapter that can report the UTC offset in effect at a given instant.
///
/// The offset is expressed in seconds, such that `local = utc + offset`.
///
/// A blanket implementation is provided for closures `Fn(i64) -> Option<i64>`, so simple
/// adapters and test fixtures can be plain functions.
pub trait OffsetAt {
    /// Returns the UTC offset in seconds in effect at UTC timestamp `utc_ts`
    /// (seconds since the Unix epoch), or `None` if it cannot be determined.
    ///
    /// The returned offset must be strictly less than 24 hours in magnitude.
    fn offset_at(&self, utc_ts: i64) -> Option<i64>;
}

impl<F: Fn(i64) -> Option<i64>> OffsetAt for F {
    fn offset_at(&self, utc_ts: i64) -> Option<i64> {
        self(utc_ts)
    }
}

/// A maximal half-open UTC interval over which the local UTC offset is constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OffsetSegment {
    /// Inclusive UTC start of the segment, in seconds since the Unix epoch.
    pub utc_start: i64,
    /// Exclusive UTC end of the segment, in seconds since the Unix epoch.
    pub utc_end: i64,
    /// Local time corresponding to `utc_start` (`utc_start + offset`).
    pub local_start: i64,
    /// Local time corresponding to `utc_end` (`utc_end + offset`).
    pub local_end: i64,
    /// The constant UTC offset of this segment, in seconds.
    pub offset: i64,
}

/// Errors returned by [`split_interval`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentError {
    /// `candidates` was not sorted in non-decreasing order.
    UnsortedCandidates,
    /// The adapter could not determine the offset at the given UTC timestamp.
    UnknownOffset {
        /// The UTC timestamp at which the offset could not be determined.
        at: i64,
    },
    /// The adapter returned an offset outside the accepted range of ±24 hours.
    InvalidOffset {
        /// The UTC timestamp at which the invalid offset was reported.
        at: i64,
        /// The offending offset, in seconds.
        offset: i64,
    },
}

impl fmt::Display for SegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            SegmentError::UnsortedCandidates => {
                f.write_str("candidate transition points are not sorted")
            }
            SegmentError::UnknownOffset { at } => {
                write!(f, "time zone adapter returned no offset at UTC timestamp {at}")
            }
            SegmentError::InvalidOffset { at, offset } => {
                write!(f, "invalid UTC offset {offset}s at UTC timestamp {at}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SegmentError {}

/// Splits the half-open UTC interval `[start, end)` into segments of constant local UTC offset.
///
/// `candidates` are the UTC timestamps (whole seconds) at which the offset may change; see the
/// [module-level documentation](self) for the full contract. Only candidates strictly inside
/// `(start, end)` can produce a split; candidates outside the interval are ignored.
///
/// # Errors
///
/// * [`SegmentError::UnsortedCandidates`] if `candidates` is not sorted in non-decreasing order.
/// * [`SegmentError::UnknownOffset`] if the adapter returns `None` for any probed instant.
/// * [`SegmentError::InvalidOffset`] if the adapter returns an offset of 24 hours or more in
///   magnitude.
pub fn split_interval<Tz: OffsetAt + ?Sized>(
    tz: &Tz,
    start: i64,
    end: i64,
    candidates: &[i64],
) -> Result<Vec<OffsetSegment>, SegmentError> {
    // Candidates must be sorted (duplicates are fine and are skipped below).
    if candidates.windows(2).any(|w| w[0] > w[1]) {
        return Err(SegmentError::UnsortedCandidates);
    }

    // A zero-length or empty interval produces no segments.
    if start >= end {
        return Ok(Vec::new());
    }

    // Query the offset at `at`, validating the adapter's answer.
    let checked_offset = |at: i64| -> Result<i64, SegmentError> {
        match tz.offset_at(at) {
            None => Err(SegmentError::UnknownOffset { at }),
            Some(offset) if offset.abs() >= MAX_OFFSET_SECS => {
                Err(SegmentError::InvalidOffset { at, offset })
            }
            Some(offset) => Ok(offset),
        }
    };

    // Collect the split points: every candidate strictly inside the interval at which the offset
    // actually changes. The offset is probed at `c - 1` and at `c`, so a transition is located to
    // its exact second, and an instant exactly at the transition uses the new offset.
    let mut splits: Vec<(i64, i64)> = Vec::new(); // (split point, offset after the split)
    let mut last_candidate = None;
    for &c in candidates {
        if c <= start || c >= end || last_candidate == Some(c) {
            continue;
        }
        last_candidate = Some(c);
        let before = checked_offset(c - 1)?;
        let after = checked_offset(c)?;
        if before != after {
            splits.push((c, after));
        }
    }

    let first_offset = checked_offset(start)?;
    let mut segments: Vec<OffsetSegment> = Vec::with_capacity(splits.len() + 1);

    let mut seg_start = start;
    let mut seg_offset = first_offset;
    for &(point, new_offset) in &splits {
        push_segment(&mut segments, seg_start, point, seg_offset);
        seg_start = point;
        seg_offset = new_offset;
    }
    push_segment(&mut segments, seg_start, end, seg_offset);

    Ok(segments)
}

/// Appends `[seg_start, seg_end)` with `offset` to `segments`, merging with the previous segment
/// when the offsets match and skipping zero-length segments.
fn push_segment(segments: &mut Vec<OffsetSegment>, seg_start: i64, seg_end: i64, offset: i64) {
    if seg_start >= seg_end {
        return;
    }
    if let Some(last) = segments.last_mut() {
        if last.offset == offset && last.utc_end == seg_start {
            last.utc_end = seg_end;
            last.local_end = seg_end + offset;
            return;
        }
    }
    segments.push(OffsetSegment {
        utc_start: seg_start,
        utc_end: seg_end,
        local_start: seg_start + offset,
        local_end: seg_end + offset,
        offset,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test time zone: offset is +1h for UTC seconds in `[10, 20)`, 0 otherwise.
    fn step_tz(utc_ts: i64) -> Option<i64> {
        Some(if (10..20).contains(&utc_ts) { 3600 } else { 0 })
    }

    #[test]
    fn splits_interval_with_cancelling_transitions() {
        // Both endpoints have offset 0, yet the interval must be cut into three segments.
        let segments = split_interval(&step_tz, 0, 30, &[10, 20]).unwrap();
        assert_eq!(
            segments,
            [
                OffsetSegment {
                    utc_start: 0,
                    utc_end: 10,
                    local_start: 0,
                    local_end: 10,
                    offset: 0,
                },
                OffsetSegment {
                    utc_start: 10,
                    utc_end: 20,
                    local_start: 10 + 3600,
                    local_end: 20 + 3600,
                    offset: 3600,
                },
                OffsetSegment {
                    utc_start: 20,
                    utc_end: 30,
                    local_start: 20,
                    local_end: 30,
                    offset: 0,
                },
            ]
        );
    }

    #[test]
    fn endpoint_at_transition_uses_new_offset() {
        // Interval starts exactly at the forward transition.
        let segments = split_interval(&step_tz, 10, 30, &[10, 20]).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].utc_start, 10);
        assert_eq!(segments[0].offset, 3600);
        assert_eq!(segments[1].utc_start, 20);
        assert_eq!(segments[1].offset, 0);

        // Interval ends exactly at the transition: the transition instant is excluded.
        let segments = split_interval(&step_tz, 0, 10, &[10, 20]).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].offset, 0);
        assert_eq!(segments[0].utc_end, 10);
    }

    #[test]
    fn duplicate_and_out_of_range_candidates_are_ignored() {
        let with_dupes = split_interval(&step_tz, 0, 30, &[-5, 10, 10, 20, 20, 20, 100]).unwrap();
        let plain = split_interval(&step_tz, 0, 30, &[10, 20]).unwrap();
        assert_eq!(with_dupes, plain);
        assert_eq!(with_dupes.len(), 3);
    }

    #[test]
    fn candidates_without_actual_change_produce_no_split() {
        // Candidates at 5 and 25 where the offset does not change.
        let segments = split_interval(&step_tz, 0, 30, &[5, 10, 20, 25]).unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments.iter().map(|s| s.offset).collect::<Vec<_>>(), [0, 3600, 0]);
    }

    #[test]
    fn constant_offset_yields_single_segment() {
        let segments = split_interval(&step_tz, 0, 5, &[10, 20]).unwrap();
        assert_eq!(
            segments,
            [OffsetSegment { utc_start: 0, utc_end: 5, local_start: 0, local_end: 5, offset: 0 }]
        );
    }

    #[test]
    fn empty_interval_yields_no_segments() {
        assert_eq!(split_interval(&step_tz, 15, 15, &[10, 20]).unwrap(), []);
        assert_eq!(split_interval(&step_tz, 20, 10, &[10, 20]).unwrap(), []);
    }

    #[test]
    fn unsorted_candidates_are_rejected() {
        assert_eq!(
            split_interval(&step_tz, 0, 30, &[20, 10]),
            Err(SegmentError::UnsortedCandidates)
        );
    }

    #[test]
    fn unknown_offset_is_rejected() {
        let tz = |_: i64| None;
        assert_eq!(
            split_interval(&tz, 0, 30, &[10, 20]),
            Err(SegmentError::UnknownOffset { at: 9 })
        );
    }

    #[test]
    fn invalid_offset_is_rejected() {
        let tz = |_: i64| Some(86_400);
        assert_eq!(
            split_interval(&tz, 0, 30, &[10, 20]),
            Err(SegmentError::InvalidOffset { at: 9, offset: 86_400 })
        );
        let tz = |_: i64| Some(-86_400);
        assert!(matches!(
            split_interval(&tz, 0, 30, &[]),
            Err(SegmentError::InvalidOffset { at: 0, offset: -86_400 })
        ));
    }

    #[test]
    fn adjacent_same_offset_segments_are_merged() {
        // The caller also lists a spurious candidate at 15 inside a constant region; no
        // zero-length or duplicate-offset segments may appear.
        let segments = split_interval(&step_tz, 0, 30, &[10, 15, 20]).unwrap();
        assert_eq!(segments.len(), 3);
        for w in segments.windows(2) {
            assert!(w[0].utc_end == w[1].utc_start);
            assert!(w[0].offset != w[1].offset);
        }
        for s in &segments {
            assert!(s.utc_start < s.utc_end);
            assert_eq!(s.local_start, s.utc_start + s.offset);
            assert_eq!(s.local_end, s.utc_end + s.offset);
        }
    }
}
