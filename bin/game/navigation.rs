//! Lightweight objective guidance on the world's curved surface.
//!
//! Distance is the straight-line distance used by the collection trigger. The
//! bearing uses wrapped surface coordinates on cylinders/tori, avoiding a
//! chord through the world. This is a steering hint, not terrain pathfinding.

use nalgebra::Vector3;
use std::{f32::consts::PI, fmt};
use vandals_and_heroes::config::WorldShape;

const EPSILON: f32 = 1e-5;

pub struct Guidance {
    pub distance: f32,
    /// Positive is right, matching the chase camera's up × forward basis.
    pub bearing_degrees: Option<f32>,
}

impl fmt::Display for Guidance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.0}m", self.distance)?;
        if let Some(angle) = self.bearing_degrees {
            if angle.abs() < 10.0 {
                write!(f, " · ahead")?;
            } else if angle.abs() >= 170.0 {
                write!(f, " · behind")?;
            } else {
                let side = if angle > 0.0 { "right" } else { "left" };
                write!(f, " · {side} {:.0}°", angle.abs())?;
            }
        }
        Ok(())
    }
}

fn unit(v: Vector3<f32>) -> Option<Vector3<f32>> {
    let length = v.norm();
    (length.is_finite() && length > EPSILON).then(|| v / length)
}

fn wrapped_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

/// Build a tangent direction toward the objective. Wrapped coordinate deltas
/// stay useful even when a chord would project to zero on the far side of a
/// cylinder/tube. On spheres the projected chord follows the great circle.
fn surface_direction(
    shape: WorldShape,
    major_radius: f32,
    position: Vector3<f32>,
    target: Vector3<f32>,
) -> Vector3<f32> {
    let radius_xy = position.x.hypot(position.y);
    let target_radius_xy = target.x.hypot(target.y);
    if radius_xy <= EPSILON || target_radius_xy <= EPSILON {
        return target - position;
    }
    let phi = position.y.atan2(position.x);
    let d_phi = wrapped_angle(target.y.atan2(target.x) - phi);
    let (sin_phi, cos_phi) = phi.sin_cos();
    let around = Vector3::new(-sin_phi, cos_phi, 0.0);
    match shape {
        WorldShape::Cylinder => {
            around * (radius_xy * d_phi) + Vector3::z() * (target.z - position.z)
        }
        WorldShape::Torus => {
            let radial = radius_xy - major_radius;
            let theta = position.z.atan2(radial);
            let d_theta = wrapped_angle(target.z.atan2(target_radius_xy - major_radius) - theta);
            let (sin_theta, cos_theta) = theta.sin_cos();
            let around_tube = Vector3::new(-sin_theta * cos_phi, -sin_theta * sin_phi, cos_theta);
            around * (radius_xy * d_phi) + around_tube * (radial.hypot(position.z) * d_theta)
        }
        WorldShape::Sphere => target - position,
    }
}

pub fn guidance(
    shape: WorldShape,
    major_radius: f32,
    position: Vector3<f32>,
    forward: Vector3<f32>,
    up: Vector3<f32>,
    target: Vector3<f32>,
) -> Guidance {
    let distance = (target - position).norm();
    let bearing_degrees = unit(up).and_then(|up| {
        let forward = unit(forward - up * forward.dot(&up))?;
        let direction = surface_direction(shape, major_radius, position, target);
        let direction = unit(direction - up * direction.dot(&up))?;
        let right = up.cross(&forward);
        Some(
            direction
                .dot(&right)
                .atan2(direction.dot(&forward))
                .to_degrees(),
        )
    });
    Guidance {
        distance,
        bearing_degrees,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 0.01, "{actual} != {expected}");
    }

    fn sphere_guidance(target: Vector3<f32>) -> Guidance {
        guidance(
            WorldShape::Sphere,
            0.0,
            Vector3::new(0.0, 10.0, 0.0),
            -Vector3::x(),
            Vector3::y(),
            target,
        )
    }

    #[test]
    fn bearing_matches_chassis_and_camera_left_right() {
        for (target, expected) in [
            (Vector3::new(-5.0, 10.0, 0.0), 0.0),
            (Vector3::new(-5.0, 10.0, 5.0), 45.0),
            (Vector3::new(-5.0, 10.0, -5.0), -45.0),
        ] {
            near(sphere_guidance(target).bearing_degrees.unwrap(), expected);
        }
        near(
            sphere_guidance(Vector3::new(5.0, 10.0, 0.0))
                .bearing_degrees
                .unwrap()
                .abs(),
            180.0,
        );
    }

    #[test]
    fn projects_pitch_and_target_height_out_of_bearing() {
        let result = guidance(
            WorldShape::Sphere,
            0.0,
            Vector3::new(0.0, 10.0, 0.0),
            Vector3::new(-1.0, 3.0, 0.0),
            Vector3::y() * 2.0,
            Vector3::new(-3.0, 22.0, 4.0),
        );
        near(result.distance, 13.0);
        near(result.bearing_degrees.unwrap(), 53.1301);
    }

    #[test]
    fn degenerate_heading_or_target_has_no_bearing() {
        let position = Vector3::new(0.0, 10.0, 0.0);
        assert!(sphere_guidance(position).bearing_degrees.is_none());
        assert!(sphere_guidance(position * 2.0).bearing_degrees.is_none());
        assert!(sphere_guidance(-position).bearing_degrees.is_none());
        for up in [Vector3::y(), Vector3::zeros()] {
            let result = guidance(
                WorldShape::Sphere,
                0.0,
                position,
                Vector3::y(),
                up,
                Vector3::new(1.0, 10.0, 0.0),
            );
            assert!(result.bearing_degrees.is_none());
        }
    }

    fn torus_point(phi: f32, theta: f32) -> Vector3<f32> {
        let ring = 100.0 + 20.0 * theta.cos();
        Vector3::new(ring * phi.cos(), ring * phi.sin(), 20.0 * theta.sin())
    }

    #[test]
    fn torus_bearing_crosses_major_seam_in_both_directions() {
        for direction in [-1.0, 1.0] {
            let phi = direction * (PI - 0.1);
            let position = torus_point(phi, PI / 2.0);
            let forward = Vector3::new(-phi.sin(), phi.cos(), 0.0);
            let result = guidance(
                WorldShape::Torus,
                100.0,
                position,
                forward,
                Vector3::z(),
                torus_point(-phi, PI / 2.0),
            );
            near(
                result.bearing_degrees.unwrap().abs(),
                if direction > 0.0 { 0.0 } else { 180.0 },
            );
        }
    }

    #[test]
    fn torus_bearing_crosses_tube_seam_in_both_directions() {
        for direction in [-1.0, 1.0] {
            let theta = direction * (PI - 0.1);
            let position = torus_point(0.0, theta);
            let up = Vector3::new(theta.cos(), 0.0, theta.sin());
            let result = guidance(
                WorldShape::Torus,
                100.0,
                position,
                Vector3::y(),
                up,
                torus_point(0.0, -theta),
            );
            near(result.bearing_degrees.unwrap(), direction * 90.0);
        }
    }

    #[test]
    fn torus_far_side_uses_surface_angles_instead_of_chord() {
        let result = guidance(
            WorldShape::Torus,
            100.0,
            torus_point(0.0, PI / 2.0),
            Vector3::y(),
            Vector3::z(),
            torus_point(PI - 0.01, PI / 2.0),
        );
        near(result.bearing_degrees.unwrap(), 0.0);
    }

    #[test]
    fn cylinder_bearing_wraps_and_preserves_axial_direction() {
        let phi = PI - 0.1;
        let position = Vector3::new(10.0 * phi.cos(), 10.0 * phi.sin(), 0.0);
        let target = Vector3::new(10.0 * phi.cos(), -10.0 * phi.sin(), 2.0);
        let result = guidance(
            WorldShape::Cylinder,
            0.0,
            position,
            Vector3::z(),
            position.normalize(),
            target,
        );
        near(result.bearing_degrees.unwrap(), -45.0);
    }

    #[test]
    fn readout_is_concise_and_handles_missing_bearing() {
        for (bearing_degrees, expected) in [
            (Some(5.0), "12m · ahead"),
            (Some(42.0), "12m · right 42°"),
            (Some(-42.0), "12m · left 42°"),
            (Some(-178.0), "12m · behind"),
            (None, "12m"),
        ] {
            let result = Guidance {
                distance: 12.4,
                bearing_degrees,
            };
            assert_eq!(result.to_string(), expected);
        }
    }
}
