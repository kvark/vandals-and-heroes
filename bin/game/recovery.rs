//! Eligibility for ordinary unstuck recovery; no mission or reward changes.

use nalgebra::{Isometry3, Vector3};
use std::time::Duration;

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
}
