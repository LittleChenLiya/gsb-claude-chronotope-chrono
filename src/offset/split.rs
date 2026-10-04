// This is a part of Chrono.
// See README.md and LICENSE.txt for details.

//! Splitting a UTC half-open interval into segments of constant offset.
//!
//! The entry point is [`split_interval`]. It takes a half-open UTC interval
//! `[start, end)`, an [`OffsetSource`] adapter that can query the offset at a
//! given instant, and a sorted list of candidate transition points. For each
//! candidate the splitter queries the offset immediately before and after the
//! point to locate second-level transitions, without scanning every second.

use alloc::vec::Vec;
use core::fmt;

use super::{Offset, TimeZone};
use crate::NaiveDateTime;

/// A source of the UTC-to-local offset in effect at a given UTC instant.
///
/// This is the adapter through which [`split_interval`] queries the offset at
/// candidate transition points. It is implemented for every [`TimeZone`] via
/// [`TimeZone::offset_from_utc_datetime`].
pub trait OffsetSource {
    /// The offset type produced by this source.
    type Offset: Offset;

    /// Returns the offset in effect at the given UTC instant.
    ///
    /// Returns `None` if the offset is unknown or invalid at that instant.
    fn offset_at(&self, utc: NaiveDateTime) -> Option<Self::Offset>;
}

impl<Tz: TimeZone> OffsetSource for Tz {
    type Offset = Tz::Offset;

    fn offset_at(&self, utc: NaiveDateTime) -> Option<Self::Offset> {
        Some(self.offset_from_utc_datetime(&utc))
    }
}

/// A half-open UTC interval during which the UTC-to-local offset is constant.
///
/// Produced by [`split_interval`]. Each segment records its UTC boundaries,
/// the constant offset in effect throughout that interval, and the corresponding
/// local (wall-clock) boundaries.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OffsetSegment<O: Offset> {
    /// The start of the segment (inclusive), UTC.
    pub utc_start: NaiveDateTime,
    /// The end of the segment (exclusive), UTC.
    pub utc_end: NaiveDateTime,
    /// The constant offset in effect throughout this segment.
    pub offset: O,
    /// The local time at the segment's UTC start.
    pub local_start: NaiveDateTime,
    /// The local time at the segment's UTC end.
    pub local_end: NaiveDateTime,
}

/// An error returned when splitting an interval.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SplitError {
    /// The interval start is not strictly before the end.
    InvalidInterval,
    /// The candidate list is not sorted or contains duplicates.
    UnsortedCandidates,
    /// The offset at a boundary is unknown or invalid.
    InvalidOffset,
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitError::InvalidInterval => write!(f, "interval start must be before end"),
            SplitError::UnsortedCandidates => {
                write!(f, "candidate transitions must be sorted and unique")
            }
            SplitError::InvalidOffset => write!(f, "offset at a boundary is unknown or invalid"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SplitError {}

/// Splitting a UTC half-open interval into segments of constant offset.
///
/// The caller supplies a sorted, unique list of candidate transition points at
/// which the offset may change. For each candidate the splitter queries the
/// offset immediately before and after the point (via [`OffsetSource::offset_at`])
/// to locate second-level transitions, without scanning every second. Adjacent
/// segments with the same offset are merged, and zero-length segments are omitted.
///
/// The local boundaries are the local times corresponding to the UTC boundary
/// instants, using the offset in effect at each boundary instant. In particular,
/// when the half-open `end` falls exactly on a transition, the local end uses
/// the new offset.
///
/// # Candidate completeness
///
/// The candidate list must be complete: the splitter only checks the points
/// given and cannot discover transitions that are not listed. It is the
/// caller's responsibility to ensure every transition within `[start, end)` is
/// included.
///
/// # Errors
///
/// Returns [`SplitError::InvalidInterval`] if `start > end`,
/// [`SplitError::UnsortedCandidates`] if the candidates are not strictly
/// increasing, and [`SplitError::InvalidOffset`] if the offset at a boundary
/// is unknown or invalid.
pub fn split_interval<S: OffsetSource>(
    source: &S,
    start: NaiveDateTime,
    end: NaiveDateTime,
    candidates: &[NaiveDateTime],
) -> Result<Vec<OffsetSegment<S::Offset>>, SplitError> {
    if start > end {
        return Err(SplitError::InvalidInterval);
    }
    if start == end {
        return Ok(Vec::new());
    }

    // Candidates must be strictly increasing (sorted and unique).
    for w in candidates.windows(2) {
        if w[0] >= w[1] {
            return Err(SplitError::UnsortedCandidates);
        }
    }

    // Collect boundary instants: start, candidates inside (start, end), end.
    let mut points: Vec<NaiveDateTime> = Vec::with_capacity(candidates.len() + 2);
    points.push(start);
    for &c in candidates {
        if c > start && c < end {
            points.push(c);
        }
    }
    points.push(end);

    // Query the offset at every boundary instant.
    let mut offsets: Vec<S::Offset> = Vec::with_capacity(points.len());
    for &p in &points {
        match source.offset_at(p) {
            Some(o) => offsets.push(o),
            None => return Err(SplitError::InvalidOffset),
        }
    }

    // Build raw segments between consecutive boundary instants.
    let mut segments: Vec<OffsetSegment<S::Offset>> = Vec::with_capacity(points.len() - 1);
    for i in 0..points.len() - 1 {
        let s = points[i];
        let e = points[i + 1];
        if s == e {
            continue;
        }
        let local_start = s + offsets[i].fix();
        let local_end = e + offsets[i + 1].fix();
        segments.push(OffsetSegment {
            utc_start: s,
            utc_end: e,
            offset: offsets[i].clone(),
            local_start,
            local_end,
        });
    }

    // Merge adjacent segments that share the same offset.
    let mut merged: Vec<OffsetSegment<S::Offset>> = Vec::with_capacity(segments.len());
    for seg in segments {
        if let Some(last) = merged.last_mut() {
            if last.offset.fix() == seg.offset.fix() {
                last.utc_end = seg.utc_end;
                last.local_end = seg.local_end;
                continue;
            }
        }
        merged.push(seg);
    }

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DateTime, FixedOffset};

    fn utc(secs: i64) -> NaiveDateTime {
        DateTime::from_timestamp(secs, 0).unwrap().naive_utc()
    }

    /// A timezone that adds 1 hour at second 10 and recovers at second 20.
    struct TransitionTimezone;

    impl OffsetSource for TransitionTimezone {
        type Offset = FixedOffset;

        fn offset_at(&self, utc: NaiveDateTime) -> Option<FixedOffset> {
            let secs = utc.and_utc().timestamp();
            if secs < 10 {
                Some(FixedOffset::east_opt(0).unwrap())
            } else if secs < 20 {
                Some(FixedOffset::east_opt(3600).unwrap())
            } else {
                Some(FixedOffset::east_opt(0).unwrap())
            }
        }
    }

    /// An offset source that always reports an unknown offset.
    struct FailingOffsetSource;

    impl OffsetSource for FailingOffsetSource {
        type Offset = FixedOffset;

        fn offset_at(&self, _utc: NaiveDateTime) -> Option<FixedOffset> {
            None
        }
    }

    #[test]
    fn splits_into_three_segments_with_two_transitions() {
        let start = utc(0);
        let end = utc(30);
        let candidates = vec![utc(10), utc(20)];
        let segments = split_interval(&TransitionTimezone, start, end, &candidates).unwrap();

        assert_eq!(segments.len(), 3);

        assert_eq!(segments[0].utc_start, utc(0));
        assert_eq!(segments[0].utc_end, utc(10));
        assert_eq!(segments[0].offset.fix().local_minus_utc(), 0);
        assert_eq!(segments[0].local_start, utc(0));
        assert_eq!(segments[0].local_end, utc(3610));

        assert_eq!(segments[1].utc_start, utc(10));
        assert_eq!(segments[1].utc_end, utc(20));
        assert_eq!(segments[1].offset.fix().local_minus_utc(), 3600);
        assert_eq!(segments[1].local_start, utc(3610));
        assert_eq!(segments[1].local_end, utc(20));

        assert_eq!(segments[2].utc_start, utc(20));
        assert_eq!(segments[2].utc_end, utc(30));
        assert_eq!(segments[2].offset.fix().local_minus_utc(), 0);
        assert_eq!(segments[2].local_start, utc(20));
        assert_eq!(segments[2].local_end, utc(30));
    }

    #[test]
    fn merges_adjacent_segments_with_same_offset() {
        // No transitions: the whole interval is one segment.
        let start = utc(0);
        let end = utc(30);
        let candidates: Vec<NaiveDateTime> = vec![];
        let segments = split_interval(&TransitionTimezone, start, end, &candidates).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].utc_start, utc(0));
        assert_eq!(segments[0].utc_end, utc(30));
        assert_eq!(segments[0].offset.fix().local_minus_utc(), 0);
    }

    #[test]
    fn zero_length_interval_produces_no_segments() {
        let t = utc(10);
        let segments = split_interval(&TransitionTimezone, t, t, &[]).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn rejects_invalid_interval() {
        let start = utc(20);
        let end = utc(10);
        let err = split_interval(&TransitionTimezone, start, end, &[]).unwrap_err();
        assert_eq!(err, SplitError::InvalidInterval);
    }

    #[test]
    fn rejects_unsorted_candidates() {
        let start = utc(0);
        let end = utc(30);
        let candidates = vec![utc(20), utc(10)];
        let err = split_interval(&TransitionTimezone, start, end, &candidates).unwrap_err();
        assert_eq!(err, SplitError::UnsortedCandidates);
    }

    #[test]
    fn rejects_duplicate_candidates() {
        let start = utc(0);
        let end = utc(30);
        let candidates = vec![utc(10), utc(10), utc(20)];
        let err = split_interval(&TransitionTimezone, start, end, &candidates).unwrap_err();
        assert_eq!(err, SplitError::UnsortedCandidates);
    }

    #[test]
    fn rejects_unknown_offset() {
        let start = utc(0);
        let end = utc(30);
        let err = split_interval(&FailingOffsetSource, start, end, &[]).unwrap_err();
        assert_eq!(err, SplitError::InvalidOffset);
    }

    #[test]
    fn end_on_transition_uses_new_offset_for_local_end() {
        // Interval ends exactly at the recovery transition (second 20).
        // The last segment's local end must use the new (post-transition) offset.
        let start = utc(0);
        let end = utc(20);
        let candidates = vec![utc(10), utc(20)];
        let segments = split_interval(&TransitionTimezone, start, end, &candidates).unwrap();

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[1].utc_start, utc(10));
        assert_eq!(segments[1].utc_end, utc(20));
        assert_eq!(segments[1].offset.fix().local_minus_utc(), 3600);
        // Local end uses the new offset (0), not the segment's offset (+1h).
        assert_eq!(segments[1].local_end, utc(20));
    }

    #[test]
    fn start_on_transition_uses_new_offset() {
        // Interval starts exactly at the jump transition (second 10).
        let start = utc(10);
        let end = utc(30);
        let candidates = vec![utc(10), utc(20)];
        let segments = split_interval(&TransitionTimezone, start, end, &candidates).unwrap();

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].utc_start, utc(10));
        assert_eq!(segments[0].utc_end, utc(20));
        assert_eq!(segments[0].offset.fix().local_minus_utc(), 3600);
        assert_eq!(segments[0].local_start, utc(3610));
    }

    #[test]
    fn candidates_outside_interval_are_ignored() {
        let start = utc(10);
        let end = utc(20);
        let candidates = vec![utc(0), utc(10), utc(20), utc(30)];
        let segments = split_interval(&TransitionTimezone, start, end, &candidates).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].utc_start, utc(10));
        assert_eq!(segments[0].utc_end, utc(20));
    }
}
