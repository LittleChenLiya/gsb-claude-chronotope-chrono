// Integration tests for the UTC interval splitting component.

use chrono::offset::{OffsetSource, SplitError, split_interval};
use chrono::{DateTime, FixedOffset, NaiveDateTime};

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

#[test]
fn interval_splits_into_three_segments() {
    let start = utc(0);
    let end = utc(30);
    let candidates = vec![utc(10), utc(20)];
    let segments = split_interval(&TransitionTimezone, start, end, &candidates).unwrap();

    // Even though both endpoints have offset 0, the middle segment differs,
    // so the interval must be split into three segments.
    assert_eq!(segments.len(), 3);

    assert_eq!(segments[0].utc_start, utc(0));
    assert_eq!(segments[0].utc_end, utc(10));
    assert_eq!(segments[0].offset.local_minus_utc(), 0);

    assert_eq!(segments[1].utc_start, utc(10));
    assert_eq!(segments[1].utc_end, utc(20));
    assert_eq!(segments[1].offset.local_minus_utc(), 3600);

    assert_eq!(segments[2].utc_start, utc(20));
    assert_eq!(segments[2].utc_end, utc(30));
    assert_eq!(segments[2].offset.local_minus_utc(), 0);
}

#[test]
fn duplicate_candidates_are_rejected() {
    let start = utc(0);
    let end = utc(30);
    let candidates = vec![utc(10), utc(10), utc(20)];
    let err = split_interval(&TransitionTimezone, start, end, &candidates).unwrap_err();
    assert_eq!(err, SplitError::UnsortedCandidates);
}

#[test]
fn unsorted_candidates_are_rejected() {
    let start = utc(0);
    let end = utc(30);
    let candidates = vec![utc(20), utc(10)];
    let err = split_interval(&TransitionTimezone, start, end, &candidates).unwrap_err();
    assert_eq!(err, SplitError::UnsortedCandidates);
}

#[test]
fn zero_length_interval_produces_no_segments() {
    let t = utc(10);
    let segments = split_interval(&TransitionTimezone, t, t, &[]).unwrap();
    assert!(segments.is_empty());
}

#[test]
fn end_on_transition_uses_new_offset() {
    let start = utc(0);
    let end = utc(20);
    let candidates = vec![utc(10), utc(20)];
    let segments = split_interval(&TransitionTimezone, start, end, &candidates).unwrap();

    assert_eq!(segments.len(), 2);
    // The local end of the last segment uses the new (post-transition) offset.
    assert_eq!(segments[1].local_end, utc(20));
}
