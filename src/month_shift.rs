//! Explicit month shifting with configurable overflow and local-time policies.
//!
//! This module provides month arithmetic that:
//!
//! - performs all calendar arithmetic on [`NaiveDate`]/[`NaiveDateTime`] first
//!   (never approximating a month as a fixed 30-day [`TimeDelta`]),
//! - binds the resulting naive local datetime to a [`TimeZone`] only afterwards,
//! - reports which policy was actually adopted via [`MonthShift`],
//! - returns `None` on overflow or when a policy rejects the result.
//!
//! The existing [`NaiveDate::checked_add_months`] is unchanged: it always clamps
//! the day to the end of the target month. This module makes the policy explicit.
//!
//! # Day overflow policies
//!
//! A month shift can produce a day that does not exist in the target month (e.g.
//! January 31 + 1 month targets February 31). [`DayOverflowPolicy`] selects what
//! to do:
//!
//! - [`Reject`](DayOverflowPolicy::Reject): return `None`.
//! - [`Clamp`](DayOverflowPolicy::Clamp): use the last day of the target month.
//! - [`Carry`](DayOverflowPolicy::Carry): carry the overflow forward into the
//!   following month (e.g. January 31 + 1 month -> March 3).
//!
//! # Local time policies
//!
//! When the resulting naive datetime is bound to a [`TimeZone`], it may fall in
//! a _fold_ (ambiguous local time) or a _gap_ (nonexistent local time).
//! [`LocalTimePolicy`] selects what to do:
//!
//! - [`Reject`](LocalTimePolicy::Reject): return `None` for folds and gaps.
//! - [`Earliest`](LocalTimePolicy::Earliest): for folds, pick the earliest
//!   instant; gaps still return `None`.
//! - [`Latest`](LocalTimePolicy::Latest): for folds, pick the latest instant;
//!   gaps still return `None`.
//!
//! # Examples
//!
//! The four date examples below use 2023-01-31 plus one month to show the three
//! day-overflow policies, and 2023-01-15 minus one month to show a negative shift
//! crossing a year boundary.
//!
//! ```
//! use chrono::month_shift::DayOverflowPolicy;
//! use chrono::{Months, NaiveDate};
//!
//! let jan_31 = NaiveDate::from_ymd_opt(2023, 1, 31).unwrap();
//!
//! // Reject: February 31 does not exist, so the shift is rejected.
//! assert_eq!(
//!     jan_31.shift_months(Months::new(1), DayOverflowPolicy::Reject),
//!     None
//! );
//!
//! // Clamp: the day is clamped to the last day of February.
//! assert_eq!(
//!     jan_31
//!         .shift_months(Months::new(1), DayOverflowPolicy::Clamp)
//!         .unwrap()
//!         .value,
//!     NaiveDate::from_ymd_opt(2023, 2, 28).unwrap()
//! );
//!
//! // Carry: the overflow is carried forward into March.
//! assert_eq!(
//!     jan_31
//!         .shift_months(Months::new(1), DayOverflowPolicy::Carry)
//!         .unwrap()
//!         .value,
//!     NaiveDate::from_ymd_opt(2023, 3, 3).unwrap()
//! );
//!
//! // A negative shift crossing a year boundary.
//! let jan_15 = NaiveDate::from_ymd_opt(2023, 1, 15).unwrap();
//! assert_eq!(
//!     jan_15
//!         .shift_months_signed(-1, DayOverflowPolicy::Clamp)
//!         .unwrap()
//!         .value,
//!     NaiveDate::from_ymd_opt(2022, 12, 15).unwrap()
//! );
//! ```
//!
//! The adopted policy is reported on the result:
//!
//! ```
//! use chrono::month_shift::DayOverflowPolicy;
//! use chrono::{Months, NaiveDate};
//!
//! let shift = NaiveDate::from_ymd_opt(2023, 1, 31)
//!     .unwrap()
//!     .shift_months(Months::new(1), DayOverflowPolicy::Carry)
//!     .unwrap();
//! assert_eq!(shift.value, NaiveDate::from_ymd_opt(2023, 3, 3).unwrap());
//! assert_eq!(shift.day_overflow, Some(DayOverflowPolicy::Carry));
//! assert_eq!(shift.local_time, None); // naive dates have no local time
//! ```
//!
//! [`TimeDelta`]: crate::TimeDelta
//!
//! # Running the tests
//!
//! The month-shift tests are small and deterministic. Run just this module with:
//!
//! ```sh
//! cargo test --lib month_shift
//! ```
//!
//! And the doc examples with:
//!
//! ```sh
//! cargo test --doc month_shift
//! ```
//!
//! The module also works without default features (no clock involved):
//!
//! ```sh
//! cargo test --lib --no-default-features month_shift
//! ```

use crate::month::Months;
use crate::naive::{NaiveDate, NaiveDateTime};
use crate::offset::LocalResult;
use crate::traits::Datelike;
use crate::try_opt;
use crate::{DateTime, TimeZone};

/// Policy for handling a day that does not exist in the destination month.
///
/// See the [module-level documentation](crate::month_shift) for examples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DayOverflowPolicy {
    /// Reject the shift: return `None`.
    Reject,
    /// Clamp the day to the last day of the destination month.
    Clamp,
    /// Carry the overflow forward into the following month.
    Carry,
}

/// Policy for resolving a local time that is ambiguous (a _fold_) or missing (a
/// _gap_).
///
/// See the [module-level documentation](crate::month_shift) for examples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LocalTimePolicy {
    /// Reject the shift: return `None` for folds and gaps.
    Reject,
    /// For folds, pick the earliest of the two instants.
    ///
    /// Gaps still return `None`.
    Earliest,
    /// For folds, pick the latest of the two instants.
    ///
    /// Gaps still return `None`.
    Latest,
}

/// The result of a successful month shift, together with the policies that were
/// actually adopted.
///
/// `day_overflow` is `Some` when the source day did not exist in the destination
/// month and the corresponding policy was applied; it is `None` when the day fit
/// exactly.
///
/// `local_time` is `Some` when the resulting local time was ambiguous (a fold)
/// and the corresponding policy selected one of the two instants; it is `None`
/// when the local time mapped to a single instant. It is always `None` for naive
/// results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MonthShift<T> {
    /// The resulting value.
    pub value: T,
    /// The day-overflow policy that was actually applied, if any.
    pub day_overflow: Option<DayOverflowPolicy>,
    /// The local-time policy that was actually applied, if any.
    pub local_time: Option<LocalTimePolicy>,
}

/// Number of days in the given month of the given year.
const fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Shifts (`year`, `month`, `day`) by `months` with the given day-overflow
/// policy.
///
/// Returns the shifted date and the policy that was actually applied, or `None`
/// on overflow or when the policy rejects the result.
const fn shift_ymd(
    year: i32,
    month: u32,
    day: u32,
    months: i32,
    policy: DayOverflowPolicy,
) -> Option<(i32, u32, u32, Option<DayOverflowPolicy>)> {
    let year_months = try_opt!(year.checked_mul(12));
    let months_since_epoch = try_opt!(year_months.checked_add(month as i32 - 1));
    let total = try_opt!(months_since_epoch.checked_add(months));
    let year = total.div_euclid(12);
    let month = total.rem_euclid(12) as u32 + 1;

    let dim = days_in_month(year, month);
    if day <= dim {
        return Some((year, month, day, None));
    }

    match policy {
        DayOverflowPolicy::Reject => None,
        DayOverflowPolicy::Clamp => Some((year, month, dim, Some(DayOverflowPolicy::Clamp))),
        DayOverflowPolicy::Carry => {
            // Carry the overflow forward into the following month: day 1 of the
            // next month plus `day - dim - 1` days.
            let extra = day - dim;
            let (ny, nm) = if month == 12 { (year + 1, 1u32) } else { (year, month + 1) };
            if extra > days_in_month(ny, nm) {
                return None;
            }
            Some((ny, nm, extra, Some(DayOverflowPolicy::Carry)))
        }
    }
}

impl NaiveDate {
    /// Shift by `months` with an explicit day-overflow policy.
    ///
    /// This performs calendar arithmetic on the naive date; it never approximates
    /// a month as a fixed 30-day duration.
    ///
    /// # Errors
    ///
    /// Returns `None` if the resulting date would be out of range, or when
    /// [`DayOverflowPolicy::Reject`] is used and the day does not exist in the
    /// destination month.
    ///
    /// # Example
    ///
    /// ```
    /// use chrono::month_shift::DayOverflowPolicy;
    /// use chrono::{Months, NaiveDate};
    ///
    /// let date = NaiveDate::from_ymd_opt(2023, 1, 31).unwrap();
    /// assert_eq!(
    ///     date.shift_months(Months::new(1), DayOverflowPolicy::Clamp).unwrap().value,
    ///     NaiveDate::from_ymd_opt(2023, 2, 28).unwrap()
    /// );
    /// ```
    #[must_use]
    pub fn shift_months(
        self,
        months: Months,
        policy: DayOverflowPolicy,
    ) -> Option<MonthShift<NaiveDate>> {
        let months = match months.0 <= i32::MAX as u32 {
            true => months.0 as i32,
            false => return None,
        };
        self.shift_months_signed(months, policy)
    }

    /// Shift by a signed number of months with an explicit day-overflow policy.
    ///
    /// Negative `months` shifts backwards, potentially crossing a year boundary.
    ///
    /// # Errors
    ///
    /// Returns `None` if the resulting date would be out of range, or when
    /// [`DayOverflowPolicy::Reject`] is used and the day does not exist in the
    /// destination month.
    ///
    /// # Example
    ///
    /// ```
    /// use chrono::month_shift::DayOverflowPolicy;
    /// use chrono::NaiveDate;
    ///
    /// let date = NaiveDate::from_ymd_opt(2023, 1, 15).unwrap();
    /// assert_eq!(
    ///     date.shift_months_signed(-1, DayOverflowPolicy::Clamp).unwrap().value,
    ///     NaiveDate::from_ymd_opt(2022, 12, 15).unwrap()
    /// );
    /// ```
    #[must_use]
    pub fn shift_months_signed(
        self,
        months: i32,
        policy: DayOverflowPolicy,
    ) -> Option<MonthShift<NaiveDate>> {
        let (year, month, day, adopted) =
            try_opt!(shift_ymd(self.year(), self.month(), self.day(), months, policy));
        let value = try_opt!(NaiveDate::from_ymd_opt(year, month, day));
        Some(MonthShift { value, day_overflow: adopted, local_time: None })
    }
}

impl NaiveDateTime {
    /// Shift by `months` with an explicit day-overflow policy.
    ///
    /// This performs calendar arithmetic on the naive local datetime; it never
    /// approximates a month as a fixed 30-day duration.
    ///
    /// # Errors
    ///
    /// Returns `None` if the resulting datetime would be out of range, or when
    /// [`DayOverflowPolicy::Reject`] is used and the day does not exist in the
    /// destination month.
    #[must_use]
    pub fn shift_months(
        self,
        months: Months,
        policy: DayOverflowPolicy,
    ) -> Option<MonthShift<NaiveDateTime>> {
        let months = match months.0 <= i32::MAX as u32 {
            true => months.0 as i32,
            false => return None,
        };
        self.shift_months_signed(months, policy)
    }

    /// Shift by a signed number of months with an explicit day-overflow policy.
    ///
    /// Negative `months` shifts backwards, potentially crossing a year boundary.
    ///
    /// # Errors
    ///
    /// Returns `None` if the resulting datetime would be out of range, or when
    /// [`DayOverflowPolicy::Reject`] is used and the day does not exist in the
    /// destination month.
    #[must_use]
    pub fn shift_months_signed(
        self,
        months: i32,
        policy: DayOverflowPolicy,
    ) -> Option<MonthShift<NaiveDateTime>> {
        let shifted = try_opt!(self.date().shift_months_signed(months, policy));
        Some(MonthShift {
            value: NaiveDateTime::new(shifted.value, self.time()),
            day_overflow: shifted.day_overflow,
            local_time: None,
        })
    }
}

impl<Tz: TimeZone> DateTime<Tz> {
    /// Shift by `months` with explicit day-overflow and local-time policies.
    ///
    /// This first performs calendar arithmetic on the naive local datetime, then
    /// binds the result to the [`TimeZone`]. It never approximates a month as a
    /// fixed 30-day duration.
    ///
    /// # Errors
    ///
    /// Returns `None` if:
    ///
    /// - the resulting date would be out of range,
    /// - [`DayOverflowPolicy::Reject`] is used and the day does not exist in the
    ///   destination month,
    /// - the resulting local time falls in a gap (a nonexistent local time), or
    /// - the resulting local time is ambiguous and [`LocalTimePolicy::Reject`] is
    ///   used.
    #[must_use]
    pub fn shift_months(
        self,
        months: Months,
        day_policy: DayOverflowPolicy,
        time_policy: LocalTimePolicy,
    ) -> Option<MonthShift<DateTime<Tz>>> {
        let months = match months.0 <= i32::MAX as u32 {
            true => months.0 as i32,
            false => return None,
        };
        self.shift_months_signed(months, day_policy, time_policy)
    }

    /// Shift by a signed number of months with explicit day-overflow and
    /// local-time policies.
    ///
    /// Negative `months` shifts backwards, potentially crossing a year boundary.
    ///
    /// This first performs calendar arithmetic on the naive local datetime, then
    /// binds the result to the [`TimeZone`]. It never approximates a month as a
    /// fixed 30-day duration.
    ///
    /// # Errors
    ///
    /// Returns `None` if:
    ///
    /// - the resulting date would be out of range,
    /// - [`DayOverflowPolicy::Reject`] is used and the day does not exist in the
    ///   destination month,
    /// - the resulting local time falls in a gap (a nonexistent local time), or
    /// - the resulting local time is ambiguous and [`LocalTimePolicy::Reject`] is
    ///   used.
    #[must_use]
    pub fn shift_months_signed(
        self,
        months: i32,
        day_policy: DayOverflowPolicy,
        time_policy: LocalTimePolicy,
    ) -> Option<MonthShift<DateTime<Tz>>> {
        // 1. Local calendar arithmetic first, on the naive local datetime.
        let naive = self.naive_local().shift_months_signed(months, day_policy)?;
        // 2. Then bind to the timezone.
        let resolved = self.timezone().from_local_datetime(&naive.value);
        let (value, adopted) = match (resolved, time_policy) {
            (LocalResult::Single(dt), _) => (dt, None),
            (LocalResult::Ambiguous(earliest, _latest), LocalTimePolicy::Earliest) => {
                (earliest, Some(LocalTimePolicy::Earliest))
            }
            (LocalResult::Ambiguous(_earliest, latest), LocalTimePolicy::Latest) => {
                (latest, Some(LocalTimePolicy::Latest))
            }
            (LocalResult::Ambiguous(..), LocalTimePolicy::Reject) => return None,
            (LocalResult::None, _) => return None,
        };
        Some(MonthShift { value, day_overflow: naive.day_overflow, local_time: adopted })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedOffset, NaiveDate, NaiveDateTime, Offset, TimeZone};

    // A fake timezone with a fold (ambiguous local time) and a gap (nonexistent
    // local time), for testing local-time policies.
    #[derive(Clone, Copy, Debug)]
    struct FakeOffset(i32);

    impl Offset for FakeOffset {
        fn fix(&self) -> FixedOffset {
            FixedOffset::east_opt(self.0).unwrap()
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct FakeTz;

    impl TimeZone for FakeTz {
        type Offset = FakeOffset;

        fn from_offset(_offset: &FakeOffset) -> FakeTz {
            FakeTz
        }

        fn offset_from_local_datetime(&self, local: &NaiveDateTime) -> LocalResult<FakeOffset> {
            // Fold: 2023-10-29 01:00..02:00 local is ambiguous (UTC+1 and UTC+0).
            let fold_start =
                NaiveDate::from_ymd_opt(2023, 10, 29).unwrap().and_hms_opt(1, 0, 0).unwrap();
            let fold_end =
                NaiveDate::from_ymd_opt(2023, 10, 29).unwrap().and_hms_opt(2, 0, 0).unwrap();
            if local >= &fold_start && local < &fold_end {
                return LocalResult::Ambiguous(FakeOffset(3600), FakeOffset(0));
            }
            // Gap: 2023-03-26 02:00..03:00 local does not exist.
            let gap_start =
                NaiveDate::from_ymd_opt(2023, 3, 26).unwrap().and_hms_opt(2, 0, 0).unwrap();
            let gap_end =
                NaiveDate::from_ymd_opt(2023, 3, 26).unwrap().and_hms_opt(3, 0, 0).unwrap();
            if local >= &gap_start && local < &gap_end {
                return LocalResult::None;
            }
            LocalResult::Single(FakeOffset(3600))
        }

        fn offset_from_utc_datetime(&self, _utc: &NaiveDateTime) -> FakeOffset {
            FakeOffset(3600)
        }

        fn offset_from_utc_date(&self, _utc: &NaiveDate) -> FakeOffset {
            FakeOffset(3600)
        }

        fn offset_from_local_date(&self, _local: &NaiveDate) -> LocalResult<FakeOffset> {
            LocalResult::Single(FakeOffset(3600))
        }
    }

    #[test]
    fn jan_31_plus_one_month_three_policies() {
        let jan_31 = NaiveDate::from_ymd_opt(2023, 1, 31).unwrap();

        // Reject: February 31 does not exist.
        assert_eq!(jan_31.shift_months(Months::new(1), DayOverflowPolicy::Reject), None);

        // Clamp: last day of February (2023 is not a leap year).
        let clamped = jan_31.shift_months(Months::new(1), DayOverflowPolicy::Clamp).unwrap();
        assert_eq!(clamped.value, NaiveDate::from_ymd_opt(2023, 2, 28).unwrap());
        assert_eq!(clamped.day_overflow, Some(DayOverflowPolicy::Clamp));
        assert_eq!(clamped.local_time, None);

        // Carry: overflow into March.
        let carried = jan_31.shift_months(Months::new(1), DayOverflowPolicy::Carry).unwrap();
        assert_eq!(carried.value, NaiveDate::from_ymd_opt(2023, 3, 3).unwrap());
        assert_eq!(carried.day_overflow, Some(DayOverflowPolicy::Carry));
        assert_eq!(carried.local_time, None);
    }

    #[test]
    fn jan_31_plus_one_month_leap_year() {
        let jan_31 = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        assert_eq!(
            jan_31.shift_months(Months::new(1), DayOverflowPolicy::Clamp).unwrap().value,
            NaiveDate::from_ymd_opt(2024, 2, 29).unwrap()
        );
        assert_eq!(
            jan_31.shift_months(Months::new(1), DayOverflowPolicy::Carry).unwrap().value,
            NaiveDate::from_ymd_opt(2024, 3, 2).unwrap()
        );
    }

    #[test]
    fn exact_day_needs_no_policy() {
        let date = NaiveDate::from_ymd_opt(2023, 1, 15).unwrap();
        let shift = date.shift_months(Months::new(1), DayOverflowPolicy::Reject).unwrap();
        assert_eq!(shift.value, NaiveDate::from_ymd_opt(2023, 2, 15).unwrap());
        assert_eq!(shift.day_overflow, None);
    }

    #[test]
    fn negative_months_cross_year() {
        let jan_15 = NaiveDate::from_ymd_opt(2023, 1, 15).unwrap();
        assert_eq!(
            jan_15.shift_months_signed(-1, DayOverflowPolicy::Clamp).unwrap().value,
            NaiveDate::from_ymd_opt(2022, 12, 15).unwrap()
        );
        assert_eq!(
            jan_15.shift_months_signed(-12, DayOverflowPolicy::Clamp).unwrap().value,
            NaiveDate::from_ymd_opt(2022, 1, 15).unwrap()
        );
        assert_eq!(
            jan_15.shift_months_signed(-24, DayOverflowPolicy::Clamp).unwrap().value,
            NaiveDate::from_ymd_opt(2021, 1, 15).unwrap()
        );
    }

    #[test]
    fn negative_months_clamp_and_carry() {
        // 2023-03-31 - 1 month -> February; clamp to 28, carry forward to March 3.
        let mar_31 = NaiveDate::from_ymd_opt(2023, 3, 31).unwrap();
        assert_eq!(
            mar_31.shift_months_signed(-1, DayOverflowPolicy::Clamp).unwrap().value,
            NaiveDate::from_ymd_opt(2023, 2, 28).unwrap()
        );
        assert_eq!(
            mar_31.shift_months_signed(-1, DayOverflowPolicy::Carry).unwrap().value,
            NaiveDate::from_ymd_opt(2023, 3, 3).unwrap()
        );
        assert_eq!(mar_31.shift_months_signed(-1, DayOverflowPolicy::Reject), None);
    }

    #[test]
    fn overflow_returns_none() {
        // Near the maximum representable date; shifting must not produce a result.
        assert_eq!(NaiveDate::MAX.shift_months(Months::new(1), DayOverflowPolicy::Clamp), None);
        assert_eq!(NaiveDate::MIN.shift_months_signed(-1, DayOverflowPolicy::Clamp), None);
        // Months larger than i32::MAX overflow.
        assert_eq!(
            NaiveDate::from_ymd_opt(2023, 1, 1)
                .unwrap()
                .shift_months(Months::new(i32::MAX as u32 + 1), DayOverflowPolicy::Clamp),
            None
        );
    }

    #[test]
    fn checked_add_months_unchanged() {
        // The ordinary checked_add_months still clamps to the month end.
        let jan_31 = NaiveDate::from_ymd_opt(2023, 1, 31).unwrap();
        assert_eq!(
            jan_31.checked_add_months(Months::new(1)),
            Some(NaiveDate::from_ymd_opt(2023, 2, 28).unwrap())
        );
        assert_eq!(
            jan_31.checked_add_months(Months::new(2)),
            Some(NaiveDate::from_ymd_opt(2023, 3, 31).unwrap())
        );
    }

    #[test]
    fn naive_datetime_keeps_time() {
        let dt = NaiveDate::from_ymd_opt(2023, 1, 31).unwrap().and_hms_opt(12, 34, 56).unwrap();
        let shifted = dt.shift_months(Months::new(1), DayOverflowPolicy::Clamp).unwrap();
        assert_eq!(shifted.value.date(), NaiveDate::from_ymd_opt(2023, 2, 28).unwrap());
        assert_eq!(shifted.value.time(), dt.time());
        assert_eq!(shifted.day_overflow, Some(DayOverflowPolicy::Clamp));
        assert_eq!(shifted.local_time, None);
    }

    #[test]
    fn fold_earliest_latest_reject() {
        let tz = FakeTz;
        // 2023-09-29 01:30 local + 1 month -> 2023-10-29 01:30 local, in the fold.
        let dt = tz.with_ymd_and_hms(2023, 9, 29, 1, 30, 0).unwrap();

        let earliest = dt
            .shift_months(Months::new(1), DayOverflowPolicy::Clamp, LocalTimePolicy::Earliest)
            .unwrap();
        assert_eq!(
            earliest.value.naive_local(),
            NaiveDate::from_ymd_opt(2023, 10, 29).unwrap().and_hms_opt(1, 30, 0).unwrap()
        );
        assert_eq!(earliest.local_time, Some(LocalTimePolicy::Earliest));
        // The earliest instant is UTC 2023-10-29 00:30.
        assert_eq!(
            earliest.value.naive_utc(),
            NaiveDate::from_ymd_opt(2023, 10, 29).unwrap().and_hms_opt(0, 30, 0).unwrap()
        );

        let latest = dt
            .shift_months(Months::new(1), DayOverflowPolicy::Clamp, LocalTimePolicy::Latest)
            .unwrap();
        assert_eq!(
            latest.value.naive_local(),
            NaiveDate::from_ymd_opt(2023, 10, 29).unwrap().and_hms_opt(1, 30, 0).unwrap()
        );
        assert_eq!(latest.local_time, Some(LocalTimePolicy::Latest));
        // The latest instant is UTC 2023-10-29 01:30.
        assert_eq!(
            latest.value.naive_utc(),
            NaiveDate::from_ymd_opt(2023, 10, 29).unwrap().and_hms_opt(1, 30, 0).unwrap()
        );

        assert_eq!(
            dt.shift_months(Months::new(1), DayOverflowPolicy::Clamp, LocalTimePolicy::Reject),
            None
        );
    }

    #[test]
    fn gap_rejected_for_all_policies() {
        let tz = FakeTz;
        // 2023-02-26 02:30 local + 1 month -> 2023-03-26 02:30 local, in the gap.
        let dt = tz.with_ymd_and_hms(2023, 2, 26, 2, 30, 0).unwrap();
        assert_eq!(
            dt.shift_months(Months::new(1), DayOverflowPolicy::Clamp, LocalTimePolicy::Reject),
            None
        );
        assert_eq!(
            dt.shift_months(Months::new(1), DayOverflowPolicy::Clamp, LocalTimePolicy::Earliest),
            None
        );
        assert_eq!(
            dt.shift_months(Months::new(1), DayOverflowPolicy::Clamp, LocalTimePolicy::Latest),
            None
        );
    }

    #[test]
    fn single_local_time_reports_no_adoption() {
        let tz = FakeTz;
        // 2023-09-15 01:30 local + 1 month -> 2023-10-15 01:30 local, unambiguous.
        let dt = tz.with_ymd_and_hms(2023, 9, 15, 1, 30, 0).unwrap();
        let shifted = dt
            .shift_months(Months::new(1), DayOverflowPolicy::Clamp, LocalTimePolicy::Reject)
            .unwrap();
        assert_eq!(shifted.local_time, None);
        assert_eq!(
            shifted.value.naive_local(),
            NaiveDate::from_ymd_opt(2023, 10, 15).unwrap().and_hms_opt(1, 30, 0).unwrap()
        );
    }

    #[test]
    fn utc_timezone_always_single() {
        let dt = crate::Utc.with_ymd_and_hms(2023, 1, 31, 12, 0, 0).unwrap();
        let shifted = dt
            .shift_months(Months::new(1), DayOverflowPolicy::Clamp, LocalTimePolicy::Reject)
            .unwrap();
        assert_eq!(
            shifted.value.naive_local(),
            NaiveDate::from_ymd_opt(2023, 2, 28).unwrap().and_hms_opt(12, 0, 0).unwrap()
        );
        assert_eq!(shifted.day_overflow, Some(DayOverflowPolicy::Clamp));
        assert_eq!(shifted.local_time, None);
    }

    #[test]
    fn negative_shift_in_fake_tz() {
        let tz = FakeTz;
        // 2023-01-15 01:30 local - 1 month -> 2022-12-15 01:30 local, unambiguous.
        let dt = tz.with_ymd_and_hms(2023, 1, 15, 1, 30, 0).unwrap();
        let shifted =
            dt.shift_months_signed(-1, DayOverflowPolicy::Clamp, LocalTimePolicy::Reject).unwrap();
        assert_eq!(
            shifted.value.naive_local(),
            NaiveDate::from_ymd_opt(2022, 12, 15).unwrap().and_hms_opt(1, 30, 0).unwrap()
        );
        assert_eq!(shifted.local_time, None);
    }
}
