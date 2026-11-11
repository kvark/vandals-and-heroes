//! Place mission targets on the same curved terrain the car can drive on.

use rapier3d::math::Vec3;
use vandals_and_heroes::{Physics, TerrainBody, config::WorldShape};

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
    /// Steepest grade on the route from the car.
    pub max_slope: f32,
    /// Ground points every ~2 m along a climbable route from the car to
    /// the beacon (empty for a straight-line fallback placement).
    pub route: Vec<Vec3>,
    /// Surface length of `route` (else the straight distance).
    pub route_length: f32,
    /// Index into `candidate_offsets` when the route search found nothing
    /// and a straight-line offset was used instead.
    pub candidate: Option<usize>,
}

impl Placement {
    pub fn climbable(&self) -> bool {
        self.max_slope <= MAX_CLIMB_SLOPE
    }

    pub fn describe(&self) -> String {
        let how = match self.candidate {
            None => format!("{} waypoints", self.route.len()),
            Some(c) => format!("straight fallback candidate {c}"),
        };
        format!(
            "{:.0} m via {how}, steepest grade {:.2} (climbable ≤ {MAX_CLIMB_SLOPE:.2})",
            self.route_length, self.max_slope
        )
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

pub fn surface_position(
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

/// Steepest grade from `point`'s ground to the ground `SLOPE_STEP` away in
/// four tangent directions: is this a flat enough spot to drop a car on?
pub fn local_grade(physics: &Physics, terrain: &TerrainBody, point: Vec3) -> Option<f32> {
    let ground = physics.terrain_surface_point(terrain, point)?;
    let up = terrain.up(ground);
    let side = up.any_orthonormal_vector();
    let height = |p: Vec3| p.distance(terrain.gravity_anchor(p));
    let center = height(ground);
    let mut grade = 0.0_f32;
    for dir in [side, -side, up.cross(side), -up.cross(side)] {
        let near = physics.terrain_surface_point(terrain, ground + dir * SLOPE_STEP)?;
        grade = grade.max((height(near) - center).abs() / SLOPE_STEP);
    }
    Some(grade)
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

/// Straight-line fallback: try rotated/rescaled offsets and keep the first
/// whose straight surface route is climbable, else the least steep.
#[allow(clippy::too_many_arguments)]
fn place_straight(
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
                route: Vec::new(),
                route_length: position.distance(origin),
                candidate: Some(candidate),
            });
        }
        if max_slope <= MAX_CLIMB_SLOPE {
            break;
        }
    }
    best
}

/// Spacing of the route-search grid over the surface.
const GRID_STEP: f32 = 0.5;
/// Route search reach along the world axis (the tube wraps in full).
const SEARCH_RANGE: f32 = 70.0;
/// Tallest bump or dip (m) across one cell the chassis clears: 0.15 m
/// wheels high-centre on Fostral's 0.3–0.8 m boulders.
const MAX_BUMP: f32 = 0.2;
/// Route points are kept every this many grid cells (~2 m).
const ROUTE_EVERY: usize = 4;

/// Regular (along, around) grid over a torus or cylinder, centred on the
/// car. Spheres have no seam-free chart here and use the fallback.
struct Chart {
    shape: WorldShape,
    major: f32,
    radius: f32,
    along0: f32,
    around0: f32,
}

impl Chart {
    fn new(terrain: &TerrainBody, ground: Vec3) -> Option<Self> {
        let radius = ground.distance(terrain.gravity_anchor(ground));
        let (along0, around0) = match terrain.shape {
            WorldShape::Sphere => return None,
            WorldShape::Cylinder => (ground.z, ground.y.atan2(ground.x)),
            WorldShape::Torus => {
                let phi = ground.y.atan2(ground.x);
                let radial = Vec3::new(phi.cos(), phi.sin(), 0.0);
                let d = ground - radial * terrain.major_radius;
                (phi, d.z.atan2(d.dot(radial)))
            }
        };
        Some(Self {
            shape: terrain.shape,
            major: terrain.major_radius,
            radius,
            along0,
            around0,
        })
    }

    /// Reference-radius point `along` metres down the world axis and
    /// `around` radians around it from the car.
    fn point(&self, along: f32, around: f32) -> Vec3 {
        let (sin, cos) = (self.around0 + around).sin_cos();
        match self.shape {
            WorldShape::Torus => {
                let phi = self.along0 + along / self.major;
                let radial = Vec3::new(phi.cos(), phi.sin(), 0.0);
                radial * (self.major + self.radius * cos) + Vec3::Z * (self.radius * sin)
            }
            _ => Vec3::new(self.radius * cos, self.radius * sin, self.along0 + along),
        }
    }
}

/// Shortest climbable routes from the car over a surface grid: a cell is
/// open only when every neighbour is within `MAX_CLIMB_SLOPE` and no bump
/// across it exceeds `MAX_BUMP`, so routes keep the car off walls and rocks.
pub struct RouteSearch {
    nu: usize,
    nv: usize,
    ground: Vec<Option<Vec3>>,
    cost: Vec<f32>,
    parent: Vec<usize>,
}

impl RouteSearch {
    /// Opposite pairs, so `chunks_exact(2)` crosses the cell.
    const NEIGHBOURS: [(isize, isize); 8] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
    ];

    pub fn new(physics: &Physics, terrain: &TerrainBody, origin: Vec3) -> Option<Self> {
        let start = physics.terrain_surface_point(terrain, origin)?;
        let chart = Chart::new(terrain, start)?;
        let half = (SEARCH_RANGE / GRID_STEP).round() as usize;
        let nu = 2 * half + 1;
        let nv = ((std::f32::consts::TAU * chart.radius / GRID_STEP).round() as usize).max(8);
        let dv = std::f32::consts::TAU / nv as f32;
        let mut ground = Vec::with_capacity(nu * nv);
        let mut height = Vec::with_capacity(nu * nv);
        for i in 0..nu {
            let along = (i as f32 - half as f32) * GRID_STEP;
            for j in 0..nv {
                let g = physics.terrain_surface_point(terrain, chart.point(along, j as f32 * dv));
                height.push(g.map_or(f32::NAN, |g| g.distance(terrain.gravity_anchor(g))));
                ground.push(g);
            }
        }
        let neighbour = |c: usize, (di, dj): (isize, isize)| -> Option<usize> {
            let i = (c / nv) as isize + di;
            if i < 0 || i >= nu as isize {
                return None;
            }
            let j = ((c % nv) as isize + dj).rem_euclid(nv as isize);
            Some(i as usize * nv + j as usize)
        };
        let grade = |a: usize, b: usize| -> f32 {
            let (Some(ga), Some(gb)) = (ground[a], ground[b]) else {
                return f32::INFINITY;
            };
            let rise = (height[a] - height[b]).abs();
            let run = (ga.distance_squared(gb) - rise * rise).max(0.0).sqrt();
            if run < 1e-3 {
                f32::INFINITY
            } else {
                rise / run
            }
        };
        // Passable: no neighbour too steep, and no bump or dip (second
        // difference across the cell) taller than the chassis clears.
        let passable: Vec<bool> = (0..nu * nv)
            .map(|c| {
                let steep = Self::NEIGHBOURS
                    .iter()
                    .any(|&d| neighbour(c, d).is_none_or(|n| grade(c, n) > MAX_CLIMB_SLOPE));
                !steep
                    && Self::NEIGHBOURS[..].chunks_exact(2).all(|pair| {
                        let (Some(a), Some(b)) = (neighbour(c, pair[0]), neighbour(c, pair[1]))
                        else {
                            return false;
                        };
                        (height[a] + height[b] - 2.0 * height[c]).abs() <= MAX_BUMP
                    })
            })
            .collect();

        // Closed cells within 1 m: routes keep to the middle of open ground
        // so a car cutting a corner does not drop into a pit beside it.
        let crowding: Vec<f32> = (0..nu * nv)
            .map(|c| {
                let (i, j) = ((c / nv) as isize, (c % nv) as isize);
                let mut closed = 0;
                for di in -2..=2_isize {
                    for dj in -2..=2_isize {
                        let ii = i + di;
                        let open = (0..nu as isize).contains(&ii)
                            && passable
                                [ii as usize * nv + (j + dj).rem_euclid(nv as isize) as usize];
                        closed += usize::from(!open);
                    }
                }
                closed as f32
            })
            .collect();

        let mut cost = vec![f32::INFINITY; nu * nv];
        let mut parent = vec![usize::MAX; nu * nv];
        let mut heap = std::collections::BinaryHeap::new();
        let key = |c: f32| std::cmp::Reverse((c * 1000.0) as u32);
        // Seed from the open cells nearest the car (within 4 m), so a car
        // wedged against a wall still finds its way out.
        let near: Vec<(usize, f32)> = ground
            .iter()
            .enumerate()
            .filter(|&(c, _)| passable[c])
            .filter_map(|(c, g)| Some((c, g.as_ref()?.distance(start))))
            .filter(|&(_, d)| d < 4.0)
            .collect();
        let closest = near.iter().map(|&(_, d)| d).fold(f32::INFINITY, f32::min);
        for &(c, d) in &near {
            if d <= closest + 1.0 {
                cost[c] = d;
                heap.push((key(d), c));
            }
        }
        while let Some((std::cmp::Reverse(k), c)) = heap.pop() {
            if k > (cost[c] * 1000.0) as u32 {
                continue;
            }
            for &d in &Self::NEIGHBOURS {
                let Some(n) = neighbour(c, d) else { continue };
                if !passable[n] {
                    continue;
                }
                let g = grade(c, n);
                // Prefer gentle, open ground: a limit-grade metre costs
                // two, a metre beside four closed cells costs two too.
                let step = ground[c].unwrap().distance(ground[n].unwrap())
                    * (1.0 + g / MAX_CLIMB_SLOPE + 0.25 * crowding[n]);
                if cost[c] + step < cost[n] {
                    cost[n] = cost[c] + step;
                    parent[n] = c;
                    heap.push((key(cost[n]), n));
                }
            }
        }
        Some(Self {
            nu,
            nv,
            ground,
            cost,
            parent,
        })
    }

    /// Ground points of every cell a climbable route reaches.
    pub fn reached(&self) -> impl Iterator<Item = (usize, Vec3)> + '_ {
        (0..self.nu * self.nv)
            .filter(|&c| self.cost[c].is_finite())
            .filter_map(|c| Some((c, self.ground[c]?)))
    }

    /// Route from the car to `cell`, plus its length and steepest grade.
    fn route(&self, terrain: &TerrainBody, cell: usize) -> (Vec<Vec3>, f32, f32) {
        let mut cells = vec![cell];
        while let Some(&c) = cells.last() {
            match self.parent[c] {
                usize::MAX => break,
                p => cells.push(p),
            }
        }
        cells.reverse();
        let points: Vec<Vec3> = cells.iter().map(|&c| self.ground[c].unwrap()).collect();
        let height = |p: Vec3| p.distance(terrain.gravity_anchor(p));
        let (mut length, mut slope) = (0.0, 0.0_f32);
        for pair in points.windows(2) {
            let rise = (height(pair[1]) - height(pair[0])).abs();
            let d = pair[0].distance(pair[1]);
            length += d;
            slope = slope.max(rise / (d * d - rise * rise).max(1e-6).sqrt());
        }
        let mut route: Vec<Vec3> = points.iter().copied().step_by(ROUTE_EVERY).collect();
        if !(points.len() - 1).is_multiple_of(ROUTE_EVERY) {
            route.push(*points.last().unwrap());
        }
        (route, length, slope)
    }
}

/// Place a beacon at the end of a route the car can actually climb.
/// Terraced maps put cliffs between most fixed offsets and the car, so
/// search the reachable region (grade ≤ `MAX_CLIMB_SLOPE`) and take the
/// reachable spot nearest the designed offset that is still a real trip.
/// Spheres, or a car boxed in by walls, fall back to straight offsets.
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
    let routed = (|| {
        let design = surface_position(physics, terrain, origin, forward, right, ahead, lateral)?;
        let start = physics.terrain_surface_point(terrain, origin)?;
        let search = RouteSearch::new(physics, terrain, origin)?;
        let distance = ahead.hypot(lateral);
        let min_straight = (0.75 * distance).max(claim_radius + 10.0);
        let detour_from = 1.5 * distance;
        let (cell, _) = search
            .reached()
            .filter(|&(_, g)| g.distance(start) >= min_straight)
            .map(|(c, g)| {
                // Nearest the design, penalising long detours.
                let detour = (search.cost[c] - detour_from).max(0.0);
                (c, g.distance(design) + 0.5 * detour)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .or_else(|| {
                log::warn!(
                    "Route search from [{:.1}, {:.1}, {:.1}]: {} cells reached, none {min_straight:.0} m away",
                    origin.x,
                    origin.y,
                    origin.z,
                    search.reached().count()
                );
                None
            })?;
        let (route, route_length, max_slope) = search.route(terrain, cell);
        let ground = *route.last()?;
        Some(Placement {
            position: ground + terrain.up(ground) * HEIGHT,
            max_slope,
            route,
            route_length,
            candidate: None,
        })
    })();
    routed.or_else(|| {
        place_straight(
            physics,
            terrain,
            origin,
            forward,
            right,
            ahead,
            lateral,
            claim_radius,
        )
    })
}

/// Climbable route from the car to the reachable spot nearest `goal`
/// (which may itself sit behind a wall). Used by the debug autopilot.
pub fn route_toward(
    physics: &Physics,
    terrain: &TerrainBody,
    origin: Vec3,
    goal: Vec3,
) -> Option<Placement> {
    let search = RouteSearch::new(physics, terrain, origin)?;
    let (cell, _) = search
        .reached()
        .map(|(c, g)| (c, g.distance(goal)))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    let (route, route_length, max_slope) = search.route(terrain, cell);
    let ground = *route.last()?;
    Some(Placement {
        position: ground + terrain.up(ground) * HEIGHT,
        max_slope,
        route,
        route_length,
        candidate: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, TAU};
    use vandals_and_heroes::config::Map;

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
        assert_eq!(placed.candidate, None);
        assert!(placed.climbable(), "{}", placed.max_slope);
        assert!(placed.route_length >= placed.position.distance(origin) - HEIGHT - 0.5);
        assert_on_surface(&physics, &terrain, placed.position);
        assert!(placed.position.distance(origin) > 14.0);
    }

    /// Headless check on the shipped map: from the default spawn and from
    /// spots the car can reach around it (where a chase may leave it), the
    /// scrap depot and the ridge cache after it must have climbable routes.
    /// Building the full Fostral TIN takes ~10 s in release.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "builds the full map TIN; run with --release"
    )]
    fn shipped_map_beacons_are_reachable_from_spawn_and_after_a_chase() {
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
        let spawn_forward: Vec3 = (spawn.rotation * crate::car_forward_local()).into();
        let place_from = |origin: Vec3, forward: Vec3, ahead, lateral, radius| {
            let up = terrain.up(origin);
            let start = std::time::Instant::now();
            let placed = place(
                &physics,
                &terrain,
                origin,
                forward,
                up.cross(forward),
                ahead,
                lateral,
                radius,
            )
            .unwrap();
            eprintln!("placed {} in {:?}", placed.describe(), start.elapsed());
            assert!(placed.climbable(), "grade {}", placed.max_slope);
            assert_eq!(placed.candidate, None);
            assert!(placed.position.distance(origin) > radius + 5.0);
            assert_on_surface(&physics, &terrain, placed.position);
            placed
        };
        let mission = |origin: Vec3, forward: Vec3| {
            let scrap = place_from(
                origin,
                forward,
                crate::SCRAP_BEACON_AHEAD,
                crate::SCRAP_BEACON_LATERAL,
                crate::SCRAP_BEACON_RADIUS,
            );
            let arrive = scrap.route[scrap.route.len() - 2];
            let heading = (scrap.route[scrap.route.len() - 1] - arrive).normalize();
            place_from(
                scrap.position,
                heading,
                crate::RIDGE_BEACON_AHEAD,
                crate::RIDGE_BEACON_LATERAL,
                crate::RIDGE_BEACON_RADIUS,
            );
        };
        let origin: Vec3 = spawn.translation.vector.into();
        mission(origin, spawn_forward);
        // Post-chase-like starts: reachable spots 8–30 m from spawn in
        // spread-out directions, each facing away from spawn.
        let search = RouteSearch::new(&physics, &terrain, origin).unwrap();
        let ground = physics.terrain_surface_point(&terrain, origin).unwrap();
        let mut starts: Vec<Vec3> = Vec::new();
        for (_, g) in search.reached() {
            let d = g.distance(ground);
            if (8.0..30.0).contains(&d) && starts.iter().all(|s| s.distance(g) > 12.0) {
                starts.push(g);
            }
        }
        assert!(starts.len() >= 4, "only {} chase spots", starts.len());
        for g in starts.into_iter().take(6) {
            let up = terrain.up(g);
            let away = g - ground;
            let forward = (away - up * away.dot(up)).normalize();
            mission(g + up * 0.5, forward);
        }
    }
}
