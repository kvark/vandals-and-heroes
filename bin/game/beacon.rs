//! Place mission targets on the same curved terrain the car can drive on.

use rapier3d::math::Vec3;
use vandals_and_heroes::{Physics, TerrainBody};

const HEIGHT: f32 = 1.5;
const STEP: f32 = 0.5;

pub fn surface_position(
    physics: &Physics,
    terrain: &TerrainBody,
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    ahead: f32,
    lateral: f32,
) -> Option<Vec3> {
    let mut pos = physics.terrain_surface_point(terrain, origin)?;
    let up = terrain.up(pos);
    // Ignore pitch/roll (and airborne height). A vertical car still has a
    // usable heading from its right axis; all placements use world up.
    let flat_forward = forward - up * forward.dot(up);
    let forward = if flat_forward.length_squared() > 1e-6 {
        flat_forward.normalize()
    } else {
        right.cross(up).normalize_or_zero()
    };
    let offset = forward * ahead + up.cross(forward) * lateral;
    let distance = offset.length();
    let steps = (distance / STEP).ceil().max(1.0) as u32;
    let step = distance / steps as f32;
    let mut heading = offset.normalize_or_zero();
    let radius = (pos - terrain.gravity_anchor(pos)).length();
    // Short tangent steps, reprojected onto the curved reference world,
    // approximate a surface walk instead of launching a 40 m straight ray
    // off a tube whose entire radius is only 10–15 m. Carry the heading
    // onto each new tangent plane rather than steering back toward the
    // original (now off-surface) direction.
    for _ in 0..steps {
        let candidate = pos + heading * step;
        let up = terrain.up(candidate);
        pos = terrain.gravity_anchor(candidate) + up * radius;
        heading = (heading - up * heading.dot(up)).normalize_or_zero();
    }
    // The reference radius only guides travel. Height comes from the actual
    // collision/render mesh, including local ridges and native/web LODs.
    let ground = physics.terrain_surface_point(terrain, pos)?;
    Some(ground + terrain.up(ground) * HEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, TAU};
    use vandals_and_heroes::config::{Map, WorldShape};

    fn flat_terrain(shape: WorldShape, alpha: u8) -> (Physics, TerrainBody, f32) {
        let config = Map {
            radius: 10.0..15.0,
            length: 80.0 * TAU,
            density: 10.0,
            shape,
        };
        let mut physics = Physics::default();
        let terrain = physics.create_terrain(&config, vec![alpha; 64 * 256], 64, 256);
        (physics, terrain, 10.0 + 5.0 * alpha as f32 / 255.0)
    }

    fn assert_on_surface(physics: &Physics, terrain: &TerrainBody, beacon: Vec3) {
        assert!(beacon.is_finite());
        let ground = physics.terrain_surface_point(terrain, beacon).unwrap();
        assert!((beacon.distance(ground) - HEIGHT).abs() < 0.002);
    }

    #[test]
    fn torus_depot_and_ridge_stay_claimable_from_the_surface() {
        let (physics, terrain, radius) = flat_terrain(WorldShape::Torus, 128);
        for phi in [0.0_f32, 3.1, -3.1] {
            let radial = Vec3::new(phi.cos(), phi.sin(), 0.0);
            let along_ring = Vec3::new(-phi.sin(), phi.cos(), 0.0);
            for theta in [0.0_f32, FRAC_PI_2, 3.1] {
                let up = radial * theta.cos() + Vec3::Z * theta.sin();
                let origin = radial * 80.0 + up * (radius + 1.0);
                for forward in [along_ring, up.cross(along_ring)] {
                    let right = up.cross(forward);
                    for (ahead, lateral) in [
                        (crate::SCRAP_BEACON_AHEAD, crate::SCRAP_BEACON_LATERAL),
                        (crate::RIDGE_BEACON_AHEAD, crate::RIDGE_BEACON_LATERAL),
                    ] {
                        let beacon = surface_position(
                            &physics, &terrain, origin, forward, right, ahead, lateral,
                        )
                        .unwrap();
                        assert_on_surface(&physics, &terrain, beacon);
                        assert!(beacon.distance(origin) > crate::SCRAP_BEACON_RADIUS);
                    }
                }
            }
        }
        assert_eq!(crate::SCRAP_BEACON_RADIUS, 14.0);
        assert_eq!(crate::RIDGE_BEACON_RADIUS, 14.0);
    }

    #[test]
    fn regression_straight_offset_misses_the_entire_torus() {
        let (physics, terrain, radius) = flat_terrain(WorldShape::Torus, 128);
        let origin = Vec3::new(80.0 + radius + 1.0, 0.0, 0.0);
        let old = origin + Vec3::Z * 42.0 - Vec3::Y * 18.0 + Vec3::X * HEIGHT;
        let old_gap = (old - terrain.gravity_anchor(old)).length() - 15.0;
        assert!(old_gap > crate::SCRAP_BEACON_RADIUS);
        let beacon =
            surface_position(&physics, &terrain, origin, Vec3::Z, -Vec3::Y, 42.0, 18.0).unwrap();
        assert_on_surface(&physics, &terrain, beacon);
    }

    #[test]
    fn placement_samples_terrain_height_and_ignores_airborne_pitch() {
        for alpha in [0, 255] {
            let (physics, terrain, radius) = flat_terrain(WorldShape::Torus, alpha);
            let origin = Vec3::new(80.0 + radius + 20.0, 0.0, 0.0);
            // Nose straight up; heading is recovered from the right axis.
            let beacon =
                surface_position(&physics, &terrain, origin, Vec3::X, Vec3::Z, 42.0, 18.0).unwrap();
            assert_on_surface(&physics, &terrain, beacon);
            let actual_radius = beacon.distance(terrain.gravity_anchor(beacon));
            assert!((actual_radius - radius - HEIGHT).abs() < 0.1);
        }
    }

    #[test]
    fn sphere_poles_and_cylinder_ends_still_have_ground() {
        for shape in [WorldShape::Sphere, WorldShape::Cylinder] {
            let (physics, terrain, radius) = flat_terrain(shape, 128);
            let origin = if shape == WorldShape::Sphere {
                Vec3::Z * (radius + 1.0)
            } else {
                Vec3::new(radius + 1.0, 0.0, 80.0 * TAU * 0.5 - 2.0)
            };
            let up = terrain.up(origin);
            let forward = if shape == WorldShape::Sphere {
                Vec3::X
            } else {
                Vec3::Z
            };
            let beacon = surface_position(
                &physics,
                &terrain,
                origin,
                forward,
                up.cross(forward),
                42.0,
                -22.0,
            )
            .unwrap();
            assert_on_surface(&physics, &terrain, beacon);
            if shape == WorldShape::Cylinder {
                assert!(beacon.z < 80.0 * TAU * 0.5 - 1.0);
            }
        }
    }
}
