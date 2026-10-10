//! Camera boom smoothing and gravity-relative framing, independent of rendering.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

pub const RADIUS: f32 = 0.3;
/// Leave room for the chassis and near plane when a ridge retracts the boom.
const MIN_VIEW_DISTANCE: f32 = 1.5;

/// Keep a finite tangent heading even when the car points straight up/down.
pub fn horizontal_forward(forward: Vector3<f32>, up: Vector3<f32>) -> Vector3<f32> {
    let tangent = forward - up * forward.dot(&up);
    tangent.try_normalize(1e-4).unwrap_or_else(|| {
        let axis = if up.z.abs() < 0.9 {
            Vector3::z()
        } else {
            Vector3::x()
        };
        (axis - up * axis.dot(&up)).normalize()
    })
}

/// Smooth outward motion, but pull inward immediately if either the desired
/// boom or the interpolated boom crosses terrain. Smoothing must not re-clip.
pub fn follow_position(
    previous: Option<Vector3<f32>>,
    desired: Vector3<f32>,
    overhead: Vector3<f32>,
    car: Vector3<f32>,
    alpha: f32,
    mut constrain: impl FnMut(Vector3<f32>) -> Vector3<f32>,
) -> Vector3<f32> {
    let mut target = constrain(desired);
    if (target - car).norm() < MIN_VIEW_DISTANCE {
        // Radial terrain has no overhangs: looking down from local up keeps
        // the car visible when the usual rear boom is squeezed into its mesh.
        // Still sweep the fallback against the real triangles.
        let fallback = constrain(overhead);
        if (fallback - car).norm_squared() > (target - car).norm_squared() {
            target = fallback;
        }
    }
    let smoothed = previous.map_or(target, |pos| pos + (target - pos) * alpha);
    let safe = constrain(smoothed);
    if (safe - car).norm() < MIN_VIEW_DISTANCE {
        // A sharp turn or new obstruction can push even the interpolated
        // boom into the chassis. Snap to the already checked external view.
        target
    } else {
        safe
    }
}

pub fn look_rotation(
    position: Vector3<f32>,
    focus: Vector3<f32>,
    up: Vector3<f32>,
    forward: Vector3<f32>,
) -> UnitQuaternion<f32> {
    let look = (focus - position).try_normalize(1e-5).unwrap_or(forward);
    let right = up
        .cross(&look)
        .try_normalize(1e-5)
        .unwrap_or_else(|| up.cross(&forward).normalize());
    let down = look.cross(&right);
    // Camera +X is right, +Y down, +Z forward.
    UnitQuaternion::from_matrix(&Matrix3::from_columns(&[right, down, look]))
}

/// Fit the entire near-plane rectangle inside the swept sphere, including
/// very wide windows, and avoid cutting the car away when the boom retracts.
pub fn near_clip(distance: f32, fov_y: f32, aspect: f32) -> f32 {
    let half_y = (0.5 * fov_y).tan();
    let sphere_limit = RADIUS / (1.0 + half_y * half_y * (1.0 + aspect * aspect)).sqrt();
    (distance * 0.25).clamp(0.01, 0.2).min(sphere_limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothing_cannot_put_camera_back_behind_wall() {
        let result = follow_position(
            Some(Vector3::new(10.0, 0.0, 0.0)),
            Vector3::new(8.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
            Vector3::zeros(),
            0.1,
            |p| Vector3::new(p.x.min(2.0), p.y, p.z),
        );
        assert_eq!(result.x, 2.0);
        let extending = follow_position(
            Some(result),
            Vector3::new(8.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
            Vector3::zeros(),
            0.1,
            |p| p,
        );
        assert!((extending.x - 2.6).abs() < 1e-5);
    }

    #[test]
    fn squeezed_boom_and_smoothing_keep_camera_outside_car() {
        let car = Vector3::zeros();
        let overhead = Vector3::y() * 3.0;
        let result = follow_position(None, Vector3::x() * 5.0, overhead, car, 0.1, |p| {
            if p.x > 0.0 { Vector3::x() * 0.2 } else { p }
        });
        assert_eq!(result, overhead);
        let across = follow_position(
            Some(-Vector3::x() * 3.0),
            Vector3::x() * 3.0,
            overhead,
            car,
            0.5,
            |p| p,
        );
        assert_eq!(across, Vector3::x() * 3.0);
    }

    #[test]
    fn vertical_chassis_has_finite_tangent_and_rotation_at_every_pole() {
        for up in [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::z()] {
            let forward = horizontal_forward(up, up);
            assert!(forward.dot(&up).abs() < 1e-5);
            assert!((forward.norm() - 1.0).abs() < 1e-5);
            for position in [up, Vector3::zeros()] {
                let rot = look_rotation(position, Vector3::zeros(), up, forward);
                assert!(rot.coords.iter().all(|v| v.is_finite()));
                assert!((rot.norm() - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn near_plane_fits_swept_sphere_and_retracted_boom() {
        for aspect in [0.5, 1.0, 2.0, 8.0, 100.0] {
            let near = near_clip(4.0, 1.0, aspect);
            let half_y = near * 0.5_f32.tan();
            assert!(
                (near * near + half_y * half_y * (1.0 + aspect * aspect)).sqrt() <= RADIUS + 1e-5
            );
        }
        assert!(near_clip(0.2, 1.0, 2.0) <= 0.05);
    }
}
