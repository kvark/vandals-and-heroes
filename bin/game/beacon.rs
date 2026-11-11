//! Place mission targets on the same curved terrain the car can drive on.

use rapier3d::math::Vec3;
use vandals_and_heroes::{Physics, TerrainBody};

const HEIGHT: f32 = 1.5;
const STEP: f32 = 0.5;

/// Steepest sustained grade (rise over run, sampled every `SLOPE_STEP`) a
/// mission route may contain. Physics alone is not the limit: wheel friction
/// 3.0 and 10 N·m motors on 0.15 m wheels out-pull the ~3 m/s² radial gravity
/// up to ~60°. What stops OxidizeMonk is geometry: 0.15 m wheels under a
/// 1.2 m wheelbase high-centre on Fostral's terrace breaks, and playtests
/// stalled on 0.8–1.0 grades. 0.7 (~35°) keeps a margin below that.
pub const MAX_CLIMB_SLOPE: f32 = 0.7;
const SLOPE_STEP: f32 = 1.0;
/// Only the route to just inside the claim radius has to be driveable; the
/// last few metres may be a wall the beacon is claimed over.
const CLAIM_MARGIN: f32 = 2.0;

pub struct Placement {
    pub position: Vec3,
    /// Steepest grade on the straight surface route from the car.
    pub max_slope: f32,
    /// Index into `candidate_offsets`; 0 is the designed placement.
    pub candidate: usize,
}

impl Placement {
    pub fn climbable(&self) -> bool {
        self.max_slope <= MAX_CLIMB_SLOPE
    }
}

/// Steepest |rise| / run between consecutive evenly spaced height samples.
pub fn max_grade(heights: &[f32], spacing: f32) -> f32 {
    heights
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs() / spacing)
        .fold(0.0, f32::max)
}

/// The designed (ahead, lateral) offset first, then the same offset rotated
/// in 20° steps alternating sides, then at shorter/longer distances.
pub fn candidate_offsets(ahead: f32, lateral: f32) -> impl Iterator<Item = (f32, f32)> {
    [1.0_f32, 0.8, 1.2, 0.65]
        .into_iter()
        .flat_map(move |scale| {
            (0..19).map(move |i| {
                let degrees = (((i + 1) / 2) * 20) as f32 * if i % 2 == 0 { 1.0 } else { -1.0 };
                let (sin, cos) = degrees.to_radians().sin_cos();
                (
                    scale * (ahead * cos - lateral * sin),
                    scale * (ahead * sin + lateral * cos),
                )
            })
        })
}

/// Reference-radius points every `STEP` from the car toward the offset.
fn walk(
    physics: &Physics,
    terrain: &TerrainBody,
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    ahead: f32,
    lateral: f32,
) -> Option<Vec<Vec3>> {
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
    let mut path = vec![pos];
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
        path.push(pos);
    }
    Some(path)
}

#[cfg(test)]
fn surface_position(
    physics: &Physics,
    terrain: &TerrainBody,
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    ahead: f32,
    lateral: f32,
) -> Option<Vec3> {
    let path = walk(physics, terrain, origin, forward, right, ahead, lateral)?;
    // The reference radius only guides travel. Height comes from the actual
    // collision/render mesh, including local ridges and native/web LODs.
    let ground = physics.terrain_surface_point(terrain, *path.last()?)?;
    Some(ground + terrain.up(ground) * HEIGHT)
}

/// Surface position plus the steepest grade on the straight route there.
#[allow(clippy::too_many_arguments)]
fn probe(
    physics: &Physics,
    terrain: &TerrainBody,
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    ahead: f32,
    lateral: f32,
    claim_radius: f32,
) -> Option<(Vec3, f32)> {
    let path = walk(physics, terrain, origin, forward, right, ahead, lateral)?;
    let ground = physics.terrain_surface_point(terrain, *path.last()?)?;
    let spacing = path.windows(2).next().map_or(STEP, |w| w[0].distance(w[1]));
    let every = ((SLOPE_STEP / spacing).round() as usize).max(1);
    let mut heights = Vec::new();
    for point in path.iter().step_by(every) {
        let below = physics.terrain_surface_point(terrain, *point)?;
        heights.push(below.distance(terrain.gravity_anchor(below)));
        if below.distance(ground) < claim_radius - CLAIM_MARGIN {
            break;
        }
    }
    let grade = max_grade(&heights, spacing * every as f32);
    Some((ground + terrain.up(ground) * HEIGHT, grade))
}

/// Place a beacon whose straight surface route from the car stays under
/// `MAX_CLIMB_SLOPE`. Terraced maps put cliffs between most fixed offsets
/// and the car, so try rotated/rescaled offsets and keep the first climbable
/// one, else the least steep.
#[allow(clippy::too_many_arguments)]
pub fn place(
    physics: &Physics,
    terrain: &TerrainBody,
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    ahead: f32,
    lateral: f32,
    claim_radius: f32,
) -> Option<Placement> {
    let mut best: Option<Placement> = None;
    for (candidate, (a, l)) in candidate_offsets(ahead, lateral).enumerate() {
        let Some((position, max_slope)) =
            probe(physics, terrain, origin, forward, right, a, l, claim_radius)
        else {
            continue;
        };
        if best.as_ref().is_none_or(|b| max_slope < b.max_slope) {
            best = Some(Placement {
                position,
                max_slope,
                candidate,
            });
        }
        if max_slope <= MAX_CLIMB_SLOPE {
            break;
        }
    }
    best
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

    #[test]
    fn grade_is_the_steepest_step() {
        assert_eq!(max_grade(&[], 1.0), 0.0);
        assert_eq!(max_grade(&[12.0], 1.0), 0.0);
        assert!((max_grade(&[12.0, 12.2, 12.1, 13.6, 13.6], 0.5) - 3.0).abs() < 1e-5);
        assert!((max_grade(&[13.0, 12.0], 2.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn candidates_start_with_the_design_and_never_get_claimable_at_spawn() {
        let all: Vec<_> = candidate_offsets(42.0, 18.0).collect();
        assert_eq!(all[0], (42.0, 18.0));
        assert_eq!(all.len(), 76);
        let design = 42.0_f32.hypot(18.0);
        for (i, &(a, l)) in all.iter().enumerate() {
            assert!(a.hypot(l) > crate::SCRAP_BEACON_RADIUS + 10.0);
            let scale = [1.0, 0.8, 1.2, 0.65][i / 19];
            assert!((a.hypot(l) - design * scale).abs() < 1e-3);
        }
        // Alternates sides: -20°, +20°, -40°, ...
        let angle =
            |(a, l): (f32, f32)| l.atan2(a).to_degrees() - 18.0_f32.atan2(42.0).to_degrees();
        assert!((angle(all[1]) + 20.0).abs() < 1e-3);
        assert!((angle(all[2]) - 20.0).abs() < 1e-3);
        assert!((angle(all[3]) + 40.0).abs() < 1e-3);
    }

    /// Torus with a 5 m plateau on both sides of a narrow ring valley: the
    /// designed lateral offset climbs a wall; a rotated one follows the valley.
    #[test]
    fn placement_avoids_a_cliff_between_car_and_beacon() {
        let (w, h) = (64usize, 256usize);
        let mut alpha = vec![255u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let theta = (x as f32 + 0.5) / w as f32 * TAU;
                if (theta - FRAC_PI_2).abs() < 0.35 {
                    alpha[y * w + x] = 0;
                }
            }
        }
        let config = Map {
            radius: 10.0..15.0,
            length: 80.0 * TAU,
            density: 10.0,
            shape: WorldShape::Torus,
        };
        let mut physics = Physics::default();
        let terrain = physics.create_terrain(&config, alpha, w as u32, h as u32);
        let origin = Vec3::new(80.0, 0.0, 11.0);
        let (forward, right) = (Vec3::Y, -Vec3::X);
        let design = probe(&physics, &terrain, origin, forward, right, 42.0, 18.0, 14.0).unwrap();
        assert!(
            design.1 > 2.0 * MAX_CLIMB_SLOPE,
            "designed offset should hit the wall: {}",
            design.1
        );
        let placed = place(&physics, &terrain, origin, forward, right, 42.0, 18.0, 14.0).unwrap();
        assert!(placed.candidate > 0);
        assert!(placed.climbable(), "{}", placed.max_slope);
        assert_on_surface(&physics, &terrain, placed.position);
        assert!(placed.position.distance(origin) > 14.0);
    }

    /// Headless check on the shipped map: from the default spawn, the scrap
    /// depot and the ridge cache after it must have climbable routes.
    /// Building the full Fostral TIN takes ~10 s in release.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "builds the full map TIN; run with --release"
    )]
    fn shipped_map_beacons_are_reachable_from_spawn() {
        let config: vandals_and_heroes::config::Config =
            ron::de::from_bytes(&std::fs::read("data/config.ron").unwrap()).unwrap();
        let dir = std::path::Path::new("data/maps").join(&config.map);
        let mut map: Map =
            ron::de::from_bytes(&std::fs::read(dir.join("map.ron")).unwrap()).unwrap();
        let png = std::fs::read(dir.join("map.png")).unwrap();
        let mut reader = png::Decoder::new(std::io::Cursor::new(png))
            .read_info()
            .unwrap();
        let mut rgba = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut rgba).unwrap();
        let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
        if map.length == 0.0 {
            map.length = TAU * map.radius.start * info.height as f32 / info.width as f32;
        }
        let mesh = vandals_and_heroes::tin::build(
            &alpha,
            info.width,
            info.height,
            &map,
            config.terrain_quality,
        );
        let mut physics = Physics::default();
        let terrain = physics.create_terrain_mesh(&map, &mesh);
        let axial = if map.shape == WorldShape::Sphere {
            0.0
        } else {
            0.1 * map.length
        };
        let spawn = crate::Game::spawn_pose(&map, map.radius.end - 0.5, axial);
        let forward: Vec3 = (spawn.rotation * crate::car_forward_local()).into();
        let right: Vec3 = (spawn.rotation * nalgebra::Vector3::z()).into();
        let place_from = |origin: Vec3, ahead, lateral, radius| {
            let start = std::time::Instant::now();
            let design = probe(
                &physics, &terrain, origin, forward, right, ahead, lateral, radius,
            );
            let placed = place(
                &physics, &terrain, origin, forward, right, ahead, lateral, radius,
            )
            .unwrap();
            eprintln!(
                "designed grade {:.2}, placed candidate {} grade {:.2} in {:?}",
                design.map_or(f32::NAN, |d| d.1),
                placed.candidate,
                placed.max_slope,
                start.elapsed()
            );
            placed
        };
        let origin: Vec3 = spawn.translation.vector.into();
        let scrap = place_from(
            origin,
            crate::SCRAP_BEACON_AHEAD,
            crate::SCRAP_BEACON_LATERAL,
            crate::SCRAP_BEACON_RADIUS,
        );
        assert!(scrap.climbable(), "scrap grade {}", scrap.max_slope);
        let ridge = place_from(
            scrap.position,
            crate::RIDGE_BEACON_AHEAD,
            crate::RIDGE_BEACON_LATERAL,
            crate::RIDGE_BEACON_RADIUS,
        );
        assert!(ridge.climbable(), "ridge grade {}", ridge.max_slope);
    }
}
