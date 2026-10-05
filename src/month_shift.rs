//! Explicit month-shifting with configurable overflow and local-time policies.
//!
//! Unlike [`DateTime::checked_add_months`] (whose behavior is fixed and unchanged by this
//! module), the functions here make every policy decision explicit:
//!
//! * The shift is performed as a **calendar** operation on the local naive date first —
//!   never as a fixed `30 * 24h` [`TimeDelta`] — and the result is only then bound to a
//!   [`TimeZone`].
//! * A day-of-month that does not exist in the target month is handled according to a
//!   [`DayOverflowPolicy`]: reject, clamp to the end of the month, or carry the surplus
//!   days into the following month.
//! * A local time that is ambiguous (repeated) or does not exist (a gap) is resolved
//!   according to a [`LocalTimePolicy`]: reject, or pick the earlier/later side.
//! * The returned [`MonthShift`] reports the policies that were *actually applied*, so
//!   callers can see whether any adjustment took place.
//! * Any arithmetic overflow (e.g. running off the representable date range) yields
//!   `None` — no result is produced.
//! * The system clock is never consulted; everything is derived from the input value.
//!
//! # Examples
//!
//! Shifting 31 January 2023 by one month under each day-overflow policy, plus a
//! negative shift that crosses a year boundary:
//!
//! ```
//! use chrono::month_shift::{
//!     shift_datetime_months, AppliedDayPolicy, DayOverflowPolicy, LocalTimePolicy,
//! };
//! use chrono::{NaiveDate, TimeZone, Utc};
//!
//! let at_noon = |y, m, d| {
//!     Utc.from_utc_datetime(&NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(12, 0, 0).unwrap())
//! };
//! let jan31 = at_noon(2023, 1, 31);
//!
//! // Example 1: Reject — 31 February does not exist, so there is no result.
//! assert!(shift_datetime_months(&jan31, 1, DayOverflowPolicy::Reject, LocalTimePolicy::Reject).is_none());
//!
//! // Example 2: Clamp — snap to the last day of February.
//! let clamped =
//!     shift_datetime_months(&jan31, 1, DayOverflowPolicy::Clamp, LocalTimePolicy::Reject).unwrap();
//! assert_eq!(clamped.datetime().naive_local().to_string(), "2023-02-28 12:00:00");
//! assert_eq!(clamped.day_policy(), AppliedDayPolicy::Clamped);
//!
//! // Example 3: Carry — the three surplus days spill over into March.
//! let carried =
//!     shift_datetime_months(&jan31, 1, DayOverflowPolicy::Carry, LocalTimePolicy::Reject).unwrap();
//! assert_eq!(carried.datetime().naive_local().to_string(), "2023-03-03 12:00:00");
//! assert_eq!(carried.day_policy(), AppliedDayPolicy::Carried);
//!
//! // Example 4: negative shifts cross year boundaries — 31 March 2023 minus two months.
//! let mar31 = at_noon(2023, 3, 31);
//! let back =
//!     shift_datetime_months(&mar31, -2, DayOverflowPolicy::Clamp, LocalTimePolicy::Reject).unwrap();
//! assert_eq!(back.datetime().naive_local().to_string(), "2023-01-31 12:00:00");
//! assert_eq!(back.day_policy(), AppliedDayPolicy::Exact);
//! ```

use crate::offset::TimeZone;
use crate::{DateTime, Datelike, Days, MappedLocalTime, NaiveDate, NaiveDateTime, TimeDelta};

/// How far (in seconds) gap resolution scans for a valid local time before giving up.
///
/// Two days covers every real-world transition, including whole-day dateline skips.
const MAX_GAP_SCAN_SECONDS: i64 = 2 * 24 * 60 * 60;

/// Policy for a day-of-month that does not exist in the target month.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DayOverflowPolicy {
    /// Refuse to produce a result: the shift returns `None`.
    Reject,
    /// Clamp the day to the last day of the target month
    /// (e.g. 31 January + 1 month → 28/29 February).
    Clamp,
    /// Carry the surplus days into the following month
    /// (e.g. 31 January + 1 month → 2/3 March).
    Carry,
}

/// Policy for local times that are ambiguous (repeated) or do not exist (a gap).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalTimePolicy {
    /// Refuse to produce a result: the shift returns `None`.
    Reject,
    /// For a repeated local time pick the earlier of the two possible instants;
    /// for a nonexistent local time pick the latest valid instant before the gap.
    Earlier,
    /// For a repeated local time pick the later of the two possible instants;
    /// for a nonexistent local time pick the earliest valid instant after the gap.
    Later,
}

/// The day-overflow handling that was actually applied by a shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AppliedDayPolicy {
    /// The day-of-month existed in the target month; no adjustment was needed.
    Exact,
    /// The day was clamped to the last day of the target month.
    Clamped,
    /// Surplus days were carried into the following month.
    Carried,
}

/// The local-time resolution that was actually applied by a shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AppliedLocalTimePolicy {
    /// The local time mapped to exactly one instant; no resolution was needed.
    Exact,
    /// The local time was repeated; the earlier instant was chosen.
    AmbiguousEarlier,
    /// The local time was repeated; the later instant was chosen.
    AmbiguousLater,
    /// The local time did not exist; the latest valid instant before the gap was chosen.
    GapEarlier,
    /// The local time did not exist; the earliest valid instant after the gap was chosen.
    GapLater,
}

/// The result of an explicit month shift, including the policies actually applied.
#[derive(Clone)]
pub struct MonthShift<Tz: TimeZone> {
    datetime: DateTime<Tz>,
    day_policy: AppliedDayPolicy,
    local_time_policy: AppliedLocalTimePolicy,
}

impl<Tz: TimeZone> MonthShift<Tz> {
    /// The shifted date-time.
    pub fn datetime(&self) -> &DateTime<Tz> {
        &self.datetime
    }

    /// The day-overflow handling that was actually applied.
    pub fn day_policy(&self) -> AppliedDayPolicy {
        self.day_policy
    }

    /// The local-time resolution that was actually applied.
    pub fn local_time_policy(&self) -> AppliedLocalTimePolicy {
        self.local_time_policy
    }

    /// Consume the result and return the shifted date-time.
    pub fn into_datetime(self) -> DateTime<Tz> {
        self.datetime
    }
}

impl<Tz: TimeZone> core::fmt::Debug for MonthShift<Tz> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MonthShift")
            .field("datetime", &self.datetime)
            .field("day_policy", &self.day_policy)
            .field("local_time_policy", &self.local_time_policy)
            .finish()
    }
}

impl<Tz: TimeZone> PartialEq for MonthShift<Tz> {
    fn eq(&self, other: &Self) -> bool {
        self.datetime == other.datetime
            && self.day_policy == other.day_policy
            && self.local_time_policy == other.local_time_policy
    }
}

impl<Tz: TimeZone> Eq for MonthShift<Tz> {}

/// Shift a [`NaiveDate`] by a number of calendar months (positive or negative).
///
/// This is a pure calendar operation: the year and month are advanced by `months`
/// (crossing year boundaries in both directions) and the day-of-month is then
/// reconciled with the target month according to `day_policy`.
///
/// Returns the shifted date together with the [`AppliedDayPolicy`] that was actually
/// applied, or `None` if the day overflows under [`DayOverflowPolicy::Reject`] or the
/// result falls outside the representable date range.
pub fn shift_date_months(
    date: NaiveDate,
    months: i32,
    day_policy: DayOverflowPolicy,
) -> Option<(NaiveDate, AppliedDayPolicy)> {
    // Work with 0-based months so that negative shifts wrap across years correctly.
    let total = date.year() as i64 * 12 + date.month0() as i64 + months as i64;
    let year = i32::try_from(total.div_euclid(12)).ok()?;
    let month = total.rem_euclid(12) as u32 + 1;

    let day = date.day();
    let last = last_day_of_month(year, month)?;
    if day <= last {
        let shifted = NaiveDate::from_ymd_opt(year, month, day)?;
        Some((shifted, AppliedDayPolicy::Exact))
    } else {
        match day_policy {
            DayOverflowPolicy::Reject => None,
            DayOverflowPolicy::Clamp => {
                let clamped = NaiveDate::from_ymd_opt(year, month, last)?;
                Some((clamped, AppliedDayPolicy::Clamped))
            }
            DayOverflowPolicy::Carry => {
                let clamped = NaiveDate::from_ymd_opt(year, month, last)?;
                let carried = clamped.checked_add_days(Days::new((day - last) as u64))?;
                Some((carried, AppliedDayPolicy::Carried))
            }
        }
    }
}

/// Shift a [`DateTime`] by a number of calendar months using explicit policies.
///
/// The local naive date is shifted first (see [`shift_date_months`]), keeping the
/// wall-clock time-of-day untouched; the resulting naive date-time is then bound to
/// the time zone of `datetime`, resolving ambiguous or nonexistent local times with
/// `local_time_policy`.
///
/// Returns a [`MonthShift`] describing the result and the policies actually applied,
/// or `None` if the day overflows under [`DayOverflowPolicy::Reject`], the local time
/// cannot be resolved under [`LocalTimePolicy::Reject`], or any arithmetic overflows
/// the representable range.
pub fn shift_datetime_months<Tz: TimeZone>(
    datetime: &DateTime<Tz>,
    months: i32,
    day_policy: DayOverflowPolicy,
    local_time_policy: LocalTimePolicy,
) -> Option<MonthShift<Tz>> {
    let naive = datetime.naive_local();
    let (date, applied_day) = shift_date_months(naive.date(), months, day_policy)?;
    let shifted = NaiveDateTime::new(date, naive.time());

    let tz = Tz::from_offset(datetime.offset());
    let (datetime, applied_local) = resolve_local(&tz, &shifted, local_time_policy)?;
    Some(MonthShift { datetime, day_policy: applied_day, local_time_policy: applied_local })
}

/// The last day of the given month, or `None` if the month itself is out of range.
fn last_day_of_month(year: i32, month: u32) -> Option<u32> {
    let first_of_next = if month == 12 {
        NaiveDate::from_ymd_opt(year.checked_add(1)?, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    };
    match first_of_next {
        Some(first) => Some(first.pred_opt()?.day()),
        None => {
            // The only in-range month without a representable first-of-next is the
            // December of the maximum year, and December always has 31 days.
            if month == 12 && NaiveDate::from_ymd_opt(year, 12, 1).is_some() {
                Some(31)
            } else {
                None
            }
        }
    }
}

/// Bind a naive date-time to a time zone under the given local-time policy.
fn resolve_local<Tz: TimeZone>(
    tz: &Tz,
    naive: &NaiveDateTime,
    policy: LocalTimePolicy,
) -> Option<(DateTime<Tz>, AppliedLocalTimePolicy)> {
    match tz.from_local_datetime(naive) {
        MappedLocalTime::Single(dt) => Some((dt, AppliedLocalTimePolicy::Exact)),
        MappedLocalTime::Ambiguous(a, b) => match policy {
            LocalTimePolicy::Reject => None,
            LocalTimePolicy::Earlier => Some((a.min(b), AppliedLocalTimePolicy::AmbiguousEarlier)),
            LocalTimePolicy::Later => Some((a.max(b), AppliedLocalTimePolicy::AmbiguousLater)),
        },
        MappedLocalTime::None => match policy {
            LocalTimePolicy::Reject => None,
            LocalTimePolicy::Earlier => {
                resolve_gap(tz, naive, -1).map(|dt| (dt, AppliedLocalTimePolicy::GapEarlier))
            }
            LocalTimePolicy::Later => {
                resolve_gap(tz, naive, 1).map(|dt| (dt, AppliedLocalTimePolicy::GapLater))
            }
        },
    }
}

/// Scan second-by-second from a nonexistent local time towards the nearest valid one.
///
/// `direction` is `-1` (towards the latest valid instant before the gap) or `1`
/// (towards the earliest valid instant after the gap). Returns `None` if no valid
/// local time is found within [`MAX_GAP_SCAN_SECONDS`] or the scan overflows.
fn resolve_gap<Tz: TimeZone>(
    tz: &Tz,
    naive: &NaiveDateTime,
    direction: i64,
) -> Option<DateTime<Tz>> {
    let mut step = 1i64;
    while step <= MAX_GAP_SCAN_SECONDS {
        let candidate = naive.checked_add_signed(TimeDelta::seconds(direction * step))?;
        match tz.from_local_datetime(&candidate) {
            MappedLocalTime::Single(dt) => return Some(dt),
            // The gap may be directly adjacent to a fold; pick the instant closest
            // to the side of the gap we are resolving towards.
            MappedLocalTime::Ambiguous(a, b) => {
                return Some(if direction < 0 { a.max(b) } else { a.min(b) });
            }
            MappedLocalTime::None => {}
        }
        step += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset::{FixedOffset, Utc};
    use crate::{NaiveDate, TimeZone};

    fn naive(y: i32, m: u32, d: u32, h: u32, min: u32, s: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(h, min, s).unwrap()
    }

    fn utc(y: i32, m: u32, d: u32, h: u32, min: u32, s: u32) -> DateTime<Utc> {
        Utc.from_utc_datetime(&naive(y, m, d, h, min, s))
    }

    /// A fake time zone with one fold and one gap, both in 2023:
    ///
    /// * winter offset UTC+1, summer offset UTC+2;
    /// * spring forward at 2023-03-26 01:00 UTC: local 02:00–03:00 does not exist;
    /// * fall back at 2023-10-29 01:00 UTC: local 02:00–03:00 happens twice.
    #[derive(Clone, Debug)]
    struct FakeDst;

    impl FakeDst {
        fn winter() -> FixedOffset {
            FixedOffset::east_opt(3600).unwrap()
        }
        fn summer() -> FixedOffset {
            FixedOffset::east_opt(7200).unwrap()
        }
        fn gap_start() -> NaiveDateTime {
            naive(2023, 3, 26, 2, 0, 0)
        }
        fn gap_end() -> NaiveDateTime {
            naive(2023, 3, 26, 3, 0, 0)
        }
        fn fold_start() -> NaiveDateTime {
            naive(2023, 10, 29, 2, 0, 0)
        }
        fn fold_end() -> NaiveDateTime {
            naive(2023, 10, 29, 3, 0, 0)
        }
    }

    impl TimeZone for FakeDst {
        type Offset = FixedOffset;

        fn from_offset(_: &FixedOffset) -> Self {
            FakeDst
        }

        fn offset_from_local_date(&self, _: &NaiveDate) -> MappedLocalTime<FixedOffset> {
            unimplemented!()
        }

        fn offset_from_local_datetime(
            &self,
            local: &NaiveDateTime,
        ) -> MappedLocalTime<FixedOffset> {
            if *local >= FakeDst::gap_start() && *local < FakeDst::gap_end() {
                MappedLocalTime::None
            } else if *local >= FakeDst::fold_start() && *local < FakeDst::fold_end() {
                MappedLocalTime::Ambiguous(FakeDst::summer(), FakeDst::winter())
            } else if *local < FakeDst::gap_start() || *local >= FakeDst::fold_end() {
                MappedLocalTime::Single(FakeDst::winter())
            } else {
                MappedLocalTime::Single(FakeDst::summer())
            }
        }

        fn offset_from_utc_date(&self, _: &NaiveDate) -> FixedOffset {
            unimplemented!()
        }

        fn offset_from_utc_datetime(&self, utc: &NaiveDateTime) -> FixedOffset {
            if *utc >= naive(2023, 3, 26, 1, 0, 0) && *utc < naive(2023, 10, 29, 1, 0, 0) {
                FakeDst::summer()
            } else {
                FakeDst::winter()
            }
        }
    }

    #[test]
    fn jan31_plus_one_month_three_policies() {
        let jan31 = utc(2023, 1, 31, 12, 0, 0);

        // Reject: 31 February does not exist — no result.
        assert!(
            shift_datetime_months(&jan31, 1, DayOverflowPolicy::Reject, LocalTimePolicy::Reject)
                .is_none()
        );

        // Clamp: snap to 28 February.
        let clamped =
            shift_datetime_months(&jan31, 1, DayOverflowPolicy::Clamp, LocalTimePolicy::Reject)
                .unwrap();
        assert_eq!(clamped.datetime().naive_local(), naive(2023, 2, 28, 12, 0, 0));
        assert_eq!(clamped.day_policy(), AppliedDayPolicy::Clamped);
        assert_eq!(clamped.local_time_policy(), AppliedLocalTimePolicy::Exact);

        // Carry: 28 February + 3 surplus days = 3 March.
        let carried =
            shift_datetime_months(&jan31, 1, DayOverflowPolicy::Carry, LocalTimePolicy::Reject)
                .unwrap();
        assert_eq!(carried.datetime().naive_local(), naive(2023, 3, 3, 12, 0, 0));
        assert_eq!(carried.day_policy(), AppliedDayPolicy::Carried);
    }

    #[test]
    fn negative_months_cross_year_boundary() {
        // 31 March 2023 minus two months lands exactly on 31 January 2023.
        let shift = shift_datetime_months(
            &utc(2023, 3, 31, 8, 30, 0),
            -2,
            DayOverflowPolicy::Clamp,
            LocalTimePolicy::Reject,
        )
        .unwrap();
        assert_eq!(shift.datetime().naive_local(), naive(2023, 1, 31, 8, 30, 0));
        assert_eq!(shift.day_policy(), AppliedDayPolicy::Exact);

        // 31 January 2023 minus one month crosses into the previous year exactly.
        let shift = shift_datetime_months(
            &utc(2023, 1, 31, 8, 30, 0),
            -1,
            DayOverflowPolicy::Reject,
            LocalTimePolicy::Reject,
        )
        .unwrap();
        assert_eq!(shift.datetime().naive_local(), naive(2022, 12, 31, 8, 30, 0));
        assert_eq!(shift.day_policy(), AppliedDayPolicy::Exact);

        // 31 May 2023 minus three months: 31 February is clamped to 28 February.
        let shift = shift_datetime_months(
            &utc(2023, 5, 31, 8, 30, 0),
            -3,
            DayOverflowPolicy::Clamp,
            LocalTimePolicy::Reject,
        )
        .unwrap();
        assert_eq!(shift.datetime().naive_local(), naive(2023, 2, 28, 8, 30, 0));
        assert_eq!(shift.day_policy(), AppliedDayPolicy::Clamped);

        // Leap day minus twelve months clamps to 28 February of the common year.
        let shift = shift_datetime_months(
            &utc(2020, 2, 29, 8, 30, 0),
            -12,
            DayOverflowPolicy::Clamp,
            LocalTimePolicy::Reject,
        )
        .unwrap();
        assert_eq!(shift.datetime().naive_local(), naive(2019, 2, 28, 8, 30, 0));
        assert_eq!(shift.day_policy(), AppliedDayPolicy::Clamped);

        // 31 March minus one month carries the surplus into March itself.
        let shift = shift_datetime_months(
            &utc(2023, 3, 31, 8, 30, 0),
            -1,
            DayOverflowPolicy::Carry,
            LocalTimePolicy::Reject,
        )
        .unwrap();
        assert_eq!(shift.datetime().naive_local(), naive(2023, 3, 3, 8, 30, 0));
        assert_eq!(shift.day_policy(), AppliedDayPolicy::Carried);
    }

    #[test]
    fn ambiguous_local_time_policies() {
        // 2023-09-29 02:30 local is unambiguous (summer time, UTC+2).
        let start = match FakeDst.from_local_datetime(&naive(2023, 9, 29, 2, 30, 0)) {
            MappedLocalTime::Single(dt) => dt,
            other => panic!("expected a single local time, got {other:?}"),
        };

        // Shifting by one month lands on 2023-10-29 02:30, which happens twice.
        let earlier =
            shift_datetime_months(&start, 1, DayOverflowPolicy::Reject, LocalTimePolicy::Earlier)
                .unwrap();
        assert_eq!(earlier.local_time_policy(), AppliedLocalTimePolicy::AmbiguousEarlier);
        assert_eq!(earlier.datetime().naive_local(), naive(2023, 10, 29, 2, 30, 0));
        // Earlier instant: still on the summer offset (UTC+2) → 00:30 UTC.
        assert_eq!(*earlier.datetime().offset(), FakeDst::summer());
        assert_eq!(earlier.datetime().naive_utc(), naive(2023, 10, 29, 0, 30, 0));

        let later =
            shift_datetime_months(&start, 1, DayOverflowPolicy::Reject, LocalTimePolicy::Later)
                .unwrap();
        assert_eq!(later.local_time_policy(), AppliedLocalTimePolicy::AmbiguousLater);
        assert_eq!(later.datetime().naive_local(), naive(2023, 10, 29, 2, 30, 0));
        // Later instant: on the winter offset (UTC+1) → 01:30 UTC.
        assert_eq!(*later.datetime().offset(), FakeDst::winter());
        assert_eq!(later.datetime().naive_utc(), naive(2023, 10, 29, 1, 30, 0));

        // Reject: an ambiguous local time produces no result.
        assert!(
            shift_datetime_months(&start, 1, DayOverflowPolicy::Reject, LocalTimePolicy::Reject)
                .is_none()
        );
    }

    #[test]
    fn nonexistent_local_time_policies() {
        // 2023-02-26 02:30 local is unambiguous (winter time, UTC+1).
        let start = match FakeDst.from_local_datetime(&naive(2023, 2, 26, 2, 30, 0)) {
            MappedLocalTime::Single(dt) => dt,
            other => panic!("expected a single local time, got {other:?}"),
        };

        // Shifting by one month lands on 2023-03-26 02:30, which does not exist.
        let earlier =
            shift_datetime_months(&start, 1, DayOverflowPolicy::Reject, LocalTimePolicy::Earlier)
                .unwrap();
        assert_eq!(earlier.local_time_policy(), AppliedLocalTimePolicy::GapEarlier);
        // Latest valid local time before the gap: 01:59:59 on the winter offset.
        assert_eq!(earlier.datetime().naive_local(), naive(2023, 3, 26, 1, 59, 59));
        assert_eq!(*earlier.datetime().offset(), FakeDst::winter());

        let later =
            shift_datetime_months(&start, 1, DayOverflowPolicy::Reject, LocalTimePolicy::Later)
                .unwrap();
        assert_eq!(later.local_time_policy(), AppliedLocalTimePolicy::GapLater);
        // Earliest valid local time after the gap: 03:00:00 on the summer offset.
        assert_eq!(later.datetime().naive_local(), naive(2023, 3, 26, 3, 0, 0));
        assert_eq!(*later.datetime().offset(), FakeDst::summer());

        // Reject: a nonexistent local time produces no result.
        assert!(
            shift_datetime_months(&start, 1, DayOverflowPolicy::Reject, LocalTimePolicy::Reject)
                .is_none()
        );
    }

    #[test]
    fn overflow_produces_no_result() {
        // Running off either end of the representable range yields no result,
        // regardless of the policies.
        for day_policy in
            [DayOverflowPolicy::Reject, DayOverflowPolicy::Clamp, DayOverflowPolicy::Carry]
        {
            assert!(shift_date_months(NaiveDate::MAX, 1, day_policy).is_none());
            assert!(shift_date_months(NaiveDate::MIN, -1, day_policy).is_none());
            assert!(
                shift_datetime_months(
                    &DateTime::<Utc>::MAX_UTC,
                    1,
                    day_policy,
                    LocalTimePolicy::Earlier
                )
                .is_none()
            );
            assert!(
                shift_datetime_months(
                    &DateTime::<Utc>::MIN_UTC,
                    -1,
                    day_policy,
                    LocalTimePolicy::Later
                )
                .is_none()
            );
        }
    }

    #[test]
    fn zero_months_and_plain_dates() {
        // A zero shift over an unambiguous time is an exact no-op.
        let dt = utc(2023, 6, 15, 10, 20, 30);
        let shift =
            shift_datetime_months(&dt, 0, DayOverflowPolicy::Reject, LocalTimePolicy::Reject)
                .unwrap();
        assert_eq!(shift.datetime(), &dt);
        assert_eq!(shift.day_policy(), AppliedDayPolicy::Exact);
        assert_eq!(shift.local_time_policy(), AppliedLocalTimePolicy::Exact);

        // Pure date shifting, leap-year February targeted by a carry.
        let (date, applied) = shift_date_months(
            NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
            1,
            DayOverflowPolicy::Carry,
        )
        .unwrap();
        // 2024 is a leap year: 29 February + 2 surplus days = 2 March.
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 3, 2).unwrap());
        assert_eq!(applied, AppliedDayPolicy::Carried);
    }
}
