//! Unstuck recovery: which poses are safe to return to, how repeated R
//! presses escalate, and when to hint. No mission or reward changes.

use nalgebra::{Isometry3, Matrix3, Rotation3, UnitQuaternion, Vector3};
use std::{collections::VecDeque, time::Duration};

pub const COOLDOWN: Duration = Duration::from_secs(2);

/// At least two wheels must be planted, with the chassis within ~37 degrees
/// of gravity-relative upright. Roof/nose/one-wheel contacts are not safe saves.
pub fn can_record_pose(pose: &Isometry3<f32>, up: Vector3<f32>, grounded_wheels: usize) -> bool {
    grounded_wheels >= 2
        && pose.translation.vector.iter().all(|v| v.is_finite())
        && (pose.rotation * Vector3::y()).dot(&up) >= 0.8
}

pub fn ready(elapsed: Option<Duration>, repeated_key: bool) -> bool {
    !repeated_key && elapsed.is_none_or(|dt| dt >= COOLDOWN)
}

/// Good poses are sampled at most this often (simulation seconds)...
pub const SAMPLE_INTERVAL: f32 = 0.5;
/// ...and only while really driving: chassis speed (m/s) and displacement
/// since the previous sample (m). Ground contact alone is not progress: a car
/// wedged nose-first into a wall still has four wheels on terrain.
pub const MIN_SPEED: f32 = 1.0;
pub const MIN_PROGRESS: f32 = 0.4;
/// ~8 s of driving history.
pub const HISTORY_LEN: usize = 16;
/// An escalated press reaches this many samples (~2 s of driving) further back.
pub const ESCALATE_STEP: usize = 4;
/// Presses this close together, or with no progress in between, escalate.
pub const ESCALATE_WINDOW: f32 = 15.0;
/// The Nth escalating press abandons history for a fresh safe placement.
pub const SAFE_PLACEMENT_PRESS: u32 = 3;
/// Throttle held below this speed for `STUCK_SECS` shows the R hint.
pub const STUCK_SPEED: f32 = 1.0;
pub const STUCK_SECS: f32 = 4.0;

#[derive(Debug, PartialEq)]
pub enum Recovery {
    Pose(Isometry3<f32>),
    SafePlacement,
}

pub struct PoseHistory {
    poses: VecDeque<Isometry3<f32>>,
    last_sample: f32,
    last_press: Option<f32>,
    presses: u32,
    progress_since_press: bool,
}

impl Default for PoseHistory {
    fn default() -> Self {
        Self {
            poses: VecDeque::with_capacity(HISTORY_LEN),
            last_sample: f32::NEG_INFINITY,
            last_press: None,
            presses: 0,
            progress_since_press: false,
        }
    }
}

impl PoseHistory {
    pub fn latest(&self) -> Option<Isometry3<f32>> {
        self.poses.back().copied()
    }

    /// `eligible` is `can_record_pose`; speed/progress are checked here.
    pub fn record(&mut self, now: f32, pose: Isometry3<f32>, speed: f32, eligible: bool) -> bool {
        if !eligible || speed < MIN_SPEED || now - self.last_sample < SAMPLE_INTERVAL {
            return false;
        }
        if self.poses.back().is_some_and(|last| {
            (pose.translation.vector - last.translation.vector).norm() < MIN_PROGRESS
        }) {
            return false;
        }
        if self.poses.len() == HISTORY_LEN {
            self.poses.pop_front();
        }
        self.poses.push_back(pose);
        self.last_sample = now;
        self.progress_since_press = true;
        true
    }

    /// First press: latest good pose. Another press within the window (or
    /// with no progress since) drops the poses that led into the trap and
    /// goes ~2 s further back; the third, or an empty history, asks for a
    /// safe placement.
    pub fn recover(&mut self, now: f32) -> Recovery {
        let escalate = self
            .last_press
            .is_some_and(|t| now - t < ESCALATE_WINDOW || !self.progress_since_press);
        self.presses = if escalate { self.presses + 1 } else { 1 };
        self.last_press = Some(now);
        self.progress_since_press = false;
        let back = if self.presses == 1 { 0 } else { ESCALATE_STEP };
        if self.presses >= SAFE_PLACEMENT_PRESS || self.poses.len() <= back {
            self.poses.clear();
            return Recovery::SafePlacement;
        }
        self.poses.truncate(self.poses.len() - back);
        Recovery::Pose(*self.poses.back().unwrap())
    }
}

#[derive(Default)]
pub struct StuckDetector {
    since: Option<f32>,
}

impl StuckDetector {
    /// True once throttle has been held without real motion for `STUCK_SECS`.
    pub fn update(&mut self, now: f32, throttle_held: bool, speed: f32) -> bool {
        if throttle_held && speed < STUCK_SPEED {
            now - *self.since.get_or_insert(now) >= STUCK_SECS
        } else {
            self.since = None;
            false
        }
    }
}

/// Upright chassis (+Y = up) whose forward (-X) faces `toward` along the
/// ground plane, with a finite fallback when `toward` is straight up/down.
pub fn upright_facing(
    position: Vector3<f32>,
    up: Vector3<f32>,
    toward: Vector3<f32>,
) -> Isometry3<f32> {
    let forward = crate::chase::horizontal_forward(toward - position, up);
    let c_x = -forward;
    let c_z = c_x.cross(&up);
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
        Matrix3::from_columns(&[c_x, up, c_z]),
    ));
    Isometry3::from_parts(position.into(), rotation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    #[test]
    fn rejects_roof_nose_and_single_wheel_contacts() {
        let mut pose = Isometry3::identity();
        assert!(can_record_pose(&pose, Vector3::y(), 2));
        assert!(!can_record_pose(&pose, Vector3::y(), 1));
        for angle in [std::f32::consts::FRAC_PI_2, std::f32::consts::PI] {
            pose.rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle);
            assert!(!can_record_pose(&pose, Vector3::y(), 4));
        }
        // Torus/sphere/cylinder bottoms must use local gravity, not global Y.
        assert!(can_record_pose(&pose, -Vector3::y(), 4));
    }

    #[test]
    fn accepts_modest_slope_but_not_steep_ledge() {
        let mut pose = Isometry3::identity();
        pose.rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 30.0_f32.to_radians());
        assert!(can_record_pose(&pose, Vector3::y(), 4));
        pose.rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 45.0_f32.to_radians());
        assert!(!can_record_pose(&pose, Vector3::y(), 4));
        pose.translation.vector.x = f32::NAN;
        assert!(!can_record_pose(&pose, Vector3::y(), 4));
    }

    #[test]
    fn held_or_rapid_repeated_recovery_is_bounded() {
        assert!(ready(None, false));
        assert!(!ready(None, true));
        assert!(!ready(Some(Duration::from_millis(1999)), false));
        assert!(ready(Some(COOLDOWN), false));
        assert!(!ready(Some(Duration::from_secs(10)), true));
    }

    fn at(x: f32) -> Isometry3<f32> {
        Isometry3::translation(x, 0.0, 0.0)
    }

    /// Drive 1 m every 0.5 s for `n` samples starting at `t`, x.
    fn drive(history: &mut PoseHistory, t: f32, x: f32, n: usize) -> (f32, f32) {
        for i in 0..n {
            assert!(history.record(t + i as f32 * 0.5, at(x + i as f32), 2.0, true));
        }
        (t + n as f32 * 0.5, x + n as f32)
    }

    #[test]
    fn only_moving_progress_is_recorded() {
        let mut history = PoseHistory::default();
        assert!(!history.record(0.0, at(0.0), 0.2, true), "parked or wedged");
        assert!(
            !history.record(0.0, at(0.0), 3.0, false),
            "not upright/grounded"
        );
        assert!(history.record(0.0, at(0.0), 3.0, true));
        assert!(!history.record(0.2, at(1.0), 3.0, true), "too soon");
        assert!(
            !history.record(1.0, at(0.1), 3.0, true),
            "spinning wheels in place"
        );
        assert!(history.record(1.0, at(1.0), 3.0, true));
        assert_eq!(history.latest(), Some(at(1.0)));
        drive(&mut history, 2.0, 2.0, HISTORY_LEN + 3);
        assert_eq!(history.poses.len(), HISTORY_LEN);
    }

    #[test]
    fn repeated_presses_escalate_back_then_to_safe_placement() {
        let mut history = PoseHistory::default();
        assert_eq!(history.recover(0.0), Recovery::SafePlacement, "no history");
        let (t, _) = drive(&mut history, 100.0, 0.0, 10);
        // Wedged after x = 9: first press returns there.
        assert_eq!(history.recover(t), Recovery::Pose(at(9.0)));
        // Stuck again right away: ~2 s further back, dropping the trap approach.
        assert_eq!(history.recover(t + 3.0), Recovery::Pose(at(5.0)));
        assert_eq!(history.recover(t + 6.0), Recovery::SafePlacement);
        assert_eq!(history.latest(), None);
    }

    #[test]
    fn no_progress_escalates_even_after_the_window() {
        let mut history = PoseHistory::default();
        let (t, x) = drive(&mut history, 0.0, 0.0, 10);
        assert_eq!(history.recover(t), Recovery::Pose(at(9.0)));
        // Waited 30 s without driving anywhere: still escalates.
        assert_eq!(history.recover(t + 30.0), Recovery::Pose(at(5.0)));
        // Real progress, then a press long after: a fresh first press.
        let (t, _) = drive(&mut history, t + 31.0, x + 10.0, 6);
        assert_eq!(history.recover(t + 20.0), Recovery::Pose(at(x + 15.0)));
    }

    #[test]
    fn stuck_hint_needs_held_throttle_without_motion() {
        let mut stuck = StuckDetector::default();
        assert!(!stuck.update(0.0, true, 0.3));
        assert!(!stuck.update(3.9, true, 0.3));
        assert!(stuck.update(4.0, true, 0.3));
        assert!(!stuck.update(4.1, false, 0.3), "released");
        assert!(!stuck.update(9.0, true, 0.3), "timer restarted");
        assert!(!stuck.update(20.0, true, 5.0), "moving");
    }

    #[test]
    fn safe_placement_is_upright_and_faces_the_target() {
        let up = Vector3::new(0.0, 0.0, 1.0);
        let pose = upright_facing(Vector3::zeros(), up, Vector3::new(10.0, 0.0, 30.0));
        assert!((pose.rotation * Vector3::y() - up).norm() < 1e-5);
        assert!((pose.rotation * -Vector3::x() - Vector3::x()).norm() < 1e-5);
        assert!(can_record_pose(&pose, up, 4));
        // Target straight overhead still yields a finite upright pose.
        let pose = upright_facing(Vector3::zeros(), up, up * 5.0);
        assert!((pose.rotation * Vector3::y() - up).norm() < 1e-5);
    }
}
