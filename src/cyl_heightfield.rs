//! A cylindrical heightmap collider — full-resolution heightfield wrapped around the Z axis.
//!
//! The heightmap is parameterized by (u, v) where u spans the circumference (theta = 2π·u/width,
//! wrapping at width) and v spans the cylinder's axis (z = -L/2 + v·L/(height-1)). For each
//! sample, the surface lies at ground_radius = lerp(radius_start, radius_end, height(u, v)) —
//! the same formula the rendering shader uses (shaders/terrain-draw.wgsl).
//!
//! Triangles are generated on-the-fly from the grid (no upfront triangulation, no BVH); the
//! grid structure itself is the spatial index. AABB queries lower to a (cu_range, cv_range)
//! cell sweep in O(touched_cells).
//!
//! A companion [`CylDispatcher`] plugs into rapier's NarrowPhase to handle
//! `CylindricalHeightField`-vs-other-shape contact manifolds by iterating only the cells
//! overlapping the other shape's AABB.

use rapier3d::math::{Pose, Real, Vec3, Vector};
use rapier3d::parry::bounding_volume::{Aabb, BoundingSphere, BoundingVolume};
use rapier3d::parry::mass_properties::MassProperties;
use rapier3d::parry::query::details::NormalConstraints;
use rapier3d::parry::query::{
    ClosestPoints, Contact, ContactManifold, ContactManifoldsWorkspace, DefaultQueryDispatcher,
    NonlinearRigidMotion, PersistentQueryDispatcher, PointProjection, PointQuery, QueryDispatcher,
    Ray, RayCast, RayIntersection, ShapeCastHit, ShapeCastOptions, Unsupported,
};
use rapier3d::parry::shape::{Cylinder, FeatureId, Shape, ShapeType, Triangle, TypedShape};
use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Clone)]
pub struct CylindricalHeightField {
    /// Per-vertex heights packed as u8 (0..255 → 0.0..1.0). heights[v * width + u].
    /// Stored in an Arc so clone_dyn is cheap.
    heights: Arc<[u8]>,
    /// Number of samples around the circumference. Wraps modulo this.
    width: u32,
    /// Number of samples along the axis. Endpoints are at v=0 and v=height-1.
    height: u32,
    radius_start: Real,
    radius_end: Real,
    length: Real,
    aabb: Aabb,
}

impl CylindricalHeightField {
    pub fn new(
        heights: Vec<u8>,
        width: u32,
        height: u32,
        radius_start: Real,
        radius_end: Real,
        length: Real,
    ) -> Self {
        assert_eq!(heights.len() as u32, width * height, "heights size mismatch");
        let r = radius_end;
        let half_len = 0.5 * length;
        let aabb = Aabb::new(
            Vec3::new(-r, -r, -half_len),
            Vec3::new(r, r, half_len),
        );
        Self {
            heights: heights.into(),
            width,
            height,
            radius_start,
            radius_end,
            length,
            aabb,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn wrap_u(&self, u: i32) -> u32 {
        let w = self.width as i32;
        (((u % w) + w) % w) as u32
    }

    #[inline]
    fn clamp_v(&self, v: i32) -> u32 {
        v.clamp(0, self.height as i32 - 1) as u32
    }

    /// Sampled height in [0, 1] at integer grid position. u wraps; v is clamped.
    #[inline]
    fn h(&self, u: i32, v: i32) -> Real {
        let uu = self.wrap_u(u);
        let vv = self.clamp_v(v);
        let idx = (vv as usize) * (self.width as usize) + uu as usize;
        self.heights[idx] as Real / 255.0
    }

    /// Surface vertex (in heightfield local frame, which is also world for our fixed terrain)
    /// at grid coords (u, v). u wraps via modulo; v is clamped.
    #[inline]
    pub fn vertex(&self, u: i32, v: i32) -> Vec3 {
        // Use the *unwrapped* u for theta so neighbour vertices stay angularly adjacent
        // (otherwise a cell straddling u=width-1→u=0 would degenerate).
        let theta = (u as Real / self.width as Real) * std::f32::consts::TAU;
        let v_clamped = v.clamp(0, self.height as i32 - 1);
        let z = -0.5 * self.length
            + (v_clamped as Real / (self.height - 1) as Real) * self.length;
        let h = self.h(u, v);
        let r = self.radius_start + h * (self.radius_end - self.radius_start);
        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }

    /// Iterate every (cell_id, triangle) that may overlap the given local-space AABB.
    /// `cell_id` is unique per (cu, cv, tri_in_cell) so the caller can use it as a manifold key.
    pub fn map_elements_in_local_aabb(
        &self,
        aabb: &Aabb,
        f: &mut dyn FnMut(u32, &Triangle),
    ) {
        let n_cells_z = (self.height - 1) as i32;
        let half_len = 0.5 * self.length;
        let cell_z = self.length / n_cells_z as Real;

        // z range
        let z_lo = aabb.mins.z;
        let z_hi = aabb.maxs.z;
        let cv_lo = (((z_lo + half_len) / cell_z).floor() as i32).max(0);
        let cv_hi = (((z_hi + half_len) / cell_z).ceil() as i32).min(n_cells_z);
        if cv_lo >= cv_hi {
            return;
        }

        // u range from XY projection — handle wrap.
        let u_cells = self.u_cells_covering_xy(aabb.mins.x, aabb.maxs.x, aabb.mins.y, aabb.maxs.y);
        if u_cells.is_empty() {
            return;
        }

        for cv in cv_lo..cv_hi {
            for &cu in &u_cells {
                let v00 = self.vertex(cu, cv);
                let v10 = self.vertex(cu + 1, cv);
                let v01 = self.vertex(cu, cv + 1);
                let v11 = self.vertex(cu + 1, cv + 1);
                // Same triangulation as standard heightfield: split along the v00–v11 diagonal.
                let t0 = Triangle::new(v00, v01, v11);
                let t1 = Triangle::new(v00, v11, v10);

                // Pack (cv, cu_mod_width, tri_in_cell) into a u32 id
                let cu_mod = self.wrap_u(cu);
                let cell_idx = (cv as u32) * self.width + cu_mod;
                f(cell_idx * 2, &t0);
                f(cell_idx * 2 + 1, &t1);
            }
        }
    }

    /// Returns the cell indices (along u, possibly negative or > width to express wrap)
    /// that cover the angular span of the XY rectangle [x_lo..x_hi] × [y_lo..y_hi].
    fn u_cells_covering_xy(&self, x_lo: Real, x_hi: Real, y_lo: Real, y_hi: Real) -> Vec<i32> {
        // If the rectangle straddles BOTH axes (i.e. contains the origin), every theta is in.
        if x_lo <= 0.0 && x_hi >= 0.0 && y_lo <= 0.0 && y_hi >= 0.0 {
            return (0..self.width as i32).collect();
        }

        let corners = [
            (x_lo, y_lo),
            (x_hi, y_lo),
            (x_lo, y_hi),
            (x_hi, y_hi),
        ];

        // Normalize thetas to [0, TAU)
        let two_pi = std::f32::consts::TAU;
        let mut thetas: Vec<f32> = corners
            .iter()
            .map(|(x, y)| {
                let t = (*y).atan2(*x);
                if t < 0.0 {
                    t + two_pi
                } else {
                    t
                }
            })
            .collect();
        thetas.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find the largest *gap* between consecutive sorted thetas (wrapping). The arc
        // not in the gap is the one that contains all the corner directions.
        let mut max_gap = -1.0_f32;
        let mut gap_idx = 0;
        for i in 0..thetas.len() {
            let next = if i + 1 == thetas.len() {
                thetas[0] + two_pi
            } else {
                thetas[i + 1]
            };
            let gap = next - thetas[i];
            if gap > max_gap {
                max_gap = gap;
                gap_idx = i;
            }
        }
        // arc_start = the theta right after the largest gap; arc_end = the theta right before.
        let arc_start = if gap_idx + 1 == thetas.len() {
            thetas[0]
        } else {
            thetas[gap_idx + 1]
        };
        let mut arc_end = thetas[gap_idx];
        if arc_end < arc_start {
            arc_end += two_pi;
        }

        let u_per_rad = self.width as f32 / two_pi;
        let cu_lo = (arc_start * u_per_rad).floor() as i32;
        let cu_hi = (arc_end * u_per_rad).ceil() as i32;

        // Dedup after wrapping. BTreeSet keeps order.
        let mut set: BTreeSet<i32> = BTreeSet::new();
        for u in cu_lo..=cu_hi {
            set.insert(self.wrap_u(u) as i32);
        }
        set.into_iter().collect()
    }

    /// Find the (cu, cv) cell whose angular sector contains the given XY direction
    /// (clamping v within bounds). Used for point queries.
    fn cell_for(&self, pt: Vec3) -> (i32, i32) {
        let two_pi = std::f32::consts::TAU;
        let theta = pt.y.atan2(pt.x);
        let theta_n = if theta < 0.0 { theta + two_pi } else { theta };
        let cu = (theta_n / two_pi * self.width as f32).floor() as i32;

        let half_len = 0.5 * self.length;
        let n_cells_z = (self.height - 1) as i32;
        let cell_z = self.length / n_cells_z as Real;
        let cv = (((pt.z + half_len) / cell_z).floor() as i32).clamp(0, n_cells_z - 1);
        (cu, cv)
    }
}

impl Shape for CylindricalHeightField {
    fn compute_local_aabb(&self) -> Aabb {
        self.aabb
    }

    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        let half_len = 0.5 * self.length;
        BoundingSphere::new(
            Vec3::ZERO,
            (self.radius_end * self.radius_end + half_len * half_len).sqrt(),
        )
    }

    fn clone_dyn(&self) -> Box<dyn Shape> {
        Box::new(self.clone())
    }

    fn scale_dyn(&self, _scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        None
    }

    fn mass_properties(&self, density: Real) -> MassProperties {
        // Pretend we're a solid cylinder of average radius so the existing radial-gravity
        // formula (which reads `terrain.mass()`) still sees a sensible mass.
        let r = 0.5 * (self.radius_start + self.radius_end);
        Cylinder::new(0.5 * self.length, r).mass_properties(density)
    }

    fn shape_type(&self) -> ShapeType {
        ShapeType::Custom
    }

    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(self)
    }

    fn ccd_thickness(&self) -> Real {
        // Smaller of the two cell dimensions (taken at the inner radius — worst case).
        let cu = std::f32::consts::TAU * self.radius_start / self.width as Real;
        let cv = self.length / (self.height - 1) as Real;
        0.5 * cu.min(cv)
    }

    fn ccd_angular_thickness(&self) -> Real {
        std::f32::consts::FRAC_PI_4
    }
}

impl PointQuery for CylindricalHeightField {
    fn project_local_point(&self, pt: Vec3, _solid: bool) -> PointProjection {
        let (cu, cv) = self.cell_for(pt);
        let v00 = self.vertex(cu, cv);
        let v10 = self.vertex(cu + 1, cv);
        let v01 = self.vertex(cu, cv + 1);
        let v11 = self.vertex(cu + 1, cv + 1);
        let t0 = Triangle::new(v00, v01, v11);
        let t1 = Triangle::new(v00, v11, v10);
        let p0 = t0.project_local_point(pt, false);
        let p1 = t1.project_local_point(pt, false);
        if (pt - p0.point).length_squared() <= (pt - p1.point).length_squared() {
            p0
        } else {
            p1
        }
    }

    fn project_local_point_and_get_feature(&self, pt: Vec3) -> (PointProjection, FeatureId) {
        (self.project_local_point(pt, true), FeatureId::Unknown)
    }
}

impl RayCast for CylindricalHeightField {
    fn cast_local_ray_and_get_normal(
        &self,
        _ray: &Ray,
        _max_time_of_impact: Real,
        _solid: bool,
    ) -> Option<RayIntersection> {
        // TODO: proper DDA through cylindrical cells. Scene-ray queries don't fire on
        // the terrain at the moment, so this is a stub.
        None
    }
}

/// Narrow-phase dispatcher that handles `CylindricalHeightField` vs any other shape by
/// streaming the cells the other shape's AABB overlaps and delegating each generated
/// triangle to the inner `DefaultQueryDispatcher`. Falls through to the default
/// implementation for every other pair.
pub struct CylDispatcher {
    inner: DefaultQueryDispatcher,
}

impl CylDispatcher {
    pub fn new() -> Self {
        Self {
            inner: DefaultQueryDispatcher,
        }
    }

    fn try_extract<'a>(g: &'a dyn Shape) -> Option<&'a CylindricalHeightField> {
        g.downcast_ref::<CylindricalHeightField>()
    }

    fn cyl_vs_shape<ManifoldData, ContactData>(
        &self,
        pos12: &Pose,
        hf: &CylindricalHeightField,
        shape2: &dyn Shape,
        prediction: Real,
        manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
        flipped: bool,
    ) where
        ManifoldData: Default + Clone,
        ContactData: Default + Copy,
    {
        // AABB of shape2 in heightfield local space, loosened by the prediction margin so
        // we don't miss contacts that are about to form within the next solver step.
        let ls_aabb2 = shape2.compute_aabb(pos12).loosened(prediction);

        manifolds.clear();

        hf.map_elements_in_local_aabb(&ls_aabb2, &mut |cell_id, triangle| {
            let (id1, id2) = if flipped {
                (0u32, cell_id)
            } else {
                (cell_id, 0u32)
            };
            let mut manifold =
                ContactManifold::<ManifoldData, ContactData>::with_data(id1, id2, ManifoldData::default());

            let tri_dyn: &dyn Shape = triangle;
            let res = if flipped {
                self.inner.contact_manifold_convex_convex(
                    &pos12.inverse(),
                    shape2,
                    tri_dyn,
                    None,
                    None,
                    prediction,
                    &mut manifold,
                )
            } else {
                self.inner.contact_manifold_convex_convex(
                    pos12,
                    tri_dyn,
                    shape2,
                    None,
                    None,
                    prediction,
                    &mut manifold,
                )
            };

            if res.is_ok() && !manifold.points.is_empty() {
                manifolds.push(manifold);
            }
        });
    }
}

impl Default for CylDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryDispatcher for CylDispatcher {
    fn intersection_test(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<bool, Unsupported> {
        self.inner.intersection_test(pos12, g1, g2)
    }

    fn distance(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<Real, Unsupported> {
        self.inner.distance(pos12, g1, g2)
    }

    fn contact(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        prediction: Real,
    ) -> Result<Option<Contact>, Unsupported> {
        self.inner.contact(pos12, g1, g2, prediction)
    }

    fn closest_points(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        max_dist: Real,
    ) -> Result<ClosestPoints, Unsupported> {
        self.inner.closest_points(pos12, g1, g2, max_dist)
    }

    fn cast_shapes(
        &self,
        pos12: &Pose,
        vel12: Vector,
        g1: &dyn Shape,
        g2: &dyn Shape,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        self.inner
            .cast_shapes(pos12, vel12, g1, g2, options)
    }

    fn cast_shapes_nonlinear(
        &self,
        motion1: &NonlinearRigidMotion,
        g1: &dyn Shape,
        motion2: &NonlinearRigidMotion,
        g2: &dyn Shape,
        start_time: Real,
        end_time: Real,
        stop_at_penetration: bool,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        self.inner.cast_shapes_nonlinear(
            motion1,
            g1,
            motion2,
            g2,
            start_time,
            end_time,
            stop_at_penetration,
        )
    }
}

impl<ManifoldData, ContactData> PersistentQueryDispatcher<ManifoldData, ContactData> for CylDispatcher
where
    ManifoldData: Default + Clone,
    ContactData: Default + Copy,
{
    fn contact_manifolds(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        prediction: Real,
        manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
        workspace: &mut Option<ContactManifoldsWorkspace>,
    ) -> Result<(), Unsupported> {
        if let Some(hf) = Self::try_extract(g1) {
            self.cyl_vs_shape::<ManifoldData, ContactData>(
                pos12, hf, g2, prediction, manifolds, false,
            );
            Ok(())
        } else if let Some(hf) = Self::try_extract(g2) {
            self.cyl_vs_shape::<ManifoldData, ContactData>(
                &pos12.inverse(),
                hf,
                g1,
                prediction,
                manifolds,
                true,
            );
            Ok(())
        } else {
            self.inner
                .contact_manifolds(pos12, g1, g2, prediction, manifolds, workspace)
        }
    }

    fn contact_manifold_convex_convex(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        normal_constraints1: Option<&dyn NormalConstraints>,
        normal_constraints2: Option<&dyn NormalConstraints>,
        prediction: Real,
        manifold: &mut ContactManifold<ManifoldData, ContactData>,
    ) -> Result<(), Unsupported> {
        self.inner.contact_manifold_convex_convex(
            pos12,
            g1,
            g2,
            normal_constraints1,
            normal_constraints2,
            prediction,
            manifold,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: u32, height: u32, alpha: u8) -> CylindricalHeightField {
        CylindricalHeightField::new(
            vec![alpha; (width * height) as usize],
            width,
            height,
            10.0,
            20.0,
            100.0,
        )
    }

    fn approx_eq(a: Vec3, b: Vec3, tol: f32) -> bool {
        (a - b).length() < tol
    }

    #[test]
    fn vertex_at_u_zero_v_zero_is_on_positive_x_axis_at_min_z() {
        let hf = flat(8, 4, 0); // alpha=0 → r = radius_start = 10
        let v = hf.vertex(0, 0);
        assert!(approx_eq(v, Vec3::new(10.0, 0.0, -50.0), 1e-4), "{:?}", v);
    }

    #[test]
    fn vertex_at_u_quarter_is_on_positive_y_axis() {
        let hf = flat(8, 4, 0);
        // u=2 of width=8 → theta = π/2 → +Y direction
        let v = hf.vertex(2, 0);
        assert!(approx_eq(v, Vec3::new(0.0, 10.0, -50.0), 1e-4), "{:?}", v);
    }

    #[test]
    fn vertex_wraps_u_modulo_in_height_lookup_but_not_in_theta() {
        let hf = flat(8, 4, 0);
        let v8 = hf.vertex(8, 1);
        let v0 = hf.vertex(0, 1);
        // 8 wraps to 0 in alpha lookup AND theta = 2π ≡ 0, so positions coincide.
        assert!(approx_eq(v8, v0, 1e-4));
    }

    #[test]
    fn vertex_clamps_v_at_top_and_bottom() {
        let hf = flat(8, 4, 0);
        let v_clamped = hf.vertex(0, 100);
        let v_top = hf.vertex(0, 3);
        assert!(approx_eq(v_clamped, v_top, 1e-4));
    }

    #[test]
    fn vertex_radius_tracks_alpha() {
        // Tag each (u, v) with a known alpha
        let width = 4;
        let height = 4;
        let mut heights = vec![0u8; (width * height) as usize];
        heights[1 * width as usize + 0] = 255; // (u=0, v=1) → alpha=1 → r=20
        heights[2 * width as usize + 0] = 128; // (u=0, v=2) → alpha≈0.5 → r≈15
        let hf = CylindricalHeightField::new(heights, width, height, 10.0, 20.0, 30.0);

        let r1 = {
            let v = hf.vertex(0, 1);
            (v.x * v.x + v.y * v.y).sqrt()
        };
        let r2 = {
            let v = hf.vertex(0, 2);
            (v.x * v.x + v.y * v.y).sqrt()
        };
        assert!((r1 - 20.0).abs() < 1e-3, "got r1={r1}");
        assert!((r2 - 15.0196).abs() < 0.1, "got r2={r2}");
    }

    #[test]
    fn u_cells_for_aabb_containing_axis_returns_all() {
        let hf = flat(8, 4, 0);
        let cells = hf.u_cells_covering_xy(-5.0, 5.0, -5.0, 5.0);
        assert_eq!(cells.len(), 8);
    }

    #[test]
    fn u_cells_for_small_aabb_far_from_axis_returns_few() {
        let hf = flat(64, 4, 0);
        // AABB at world (10, 0) ± 0.1
        let cells = hf.u_cells_covering_xy(9.9, 10.1, -0.1, 0.1);
        // cell 0 covers [0, 2π/64] ≈ [0, 0.098]; the AABB's angular span is ≈ ±0.01
        // so we expect at most a handful of cells around 0 (with wrap).
        assert!(cells.len() <= 4, "got {} cells: {:?}", cells.len(), cells);
        // Must include cell 0 (or the wrap-around neighbor 63).
        assert!(cells.contains(&0) || cells.contains(&63), "{:?}", cells);
    }

    #[test]
    fn u_cells_for_aabb_straddling_positive_x_wraps_around_zero() {
        let hf = flat(64, 4, 0);
        // AABB at world (10, 0) with y straddling 0 — theta span crosses 0/2π.
        let cells = hf.u_cells_covering_xy(5.0, 15.0, -0.5, 0.5);
        // Should include both small-u cells (near 0) and large-u cells (near 63).
        assert!(cells.iter().any(|&u| u < 4), "missing low-u cells: {:?}", cells);
        assert!(cells.iter().any(|&u| u > 60), "missing high-u cells: {:?}", cells);
    }

    #[test]
    fn u_cells_for_aabb_near_pi_does_not_wrap() {
        let hf = flat(64, 4, 0);
        // AABB at world (-10, 0) — theta ≈ π. cells should cluster around u = 32.
        let cells = hf.u_cells_covering_xy(-10.5, -9.5, -0.5, 0.5);
        assert!(cells.iter().all(|&u| (24..40).contains(&u)), "{:?}", cells);
        assert!(cells.contains(&32));
    }

    #[test]
    fn map_elements_visits_some_triangles_inside_envelope() {
        let hf = flat(64, 32, 128); // ground at r ≈ 15
        // AABB straddling the surface near theta=0
        let aabb = Aabb::new(Vec3::new(14.5, -0.5, -1.0), Vec3::new(15.5, 0.5, 1.0));
        let mut count = 0;
        hf.map_elements_in_local_aabb(&aabb, &mut |_id, _tri| count += 1);
        assert!(count > 0, "expected at least one triangle, got {count}");
        // Bounded: only a few cells near (theta=0, z≈0) and 2 tris each.
        assert!(count < 50, "expected far fewer than 50 triangles, got {count}");
    }

    #[test]
    fn map_elements_empty_for_aabb_below_v_range() {
        let hf = flat(64, 32, 0);
        let half = 50.0;
        // AABB entirely below v=0 (z < -length/2)
        let aabb = Aabb::new(
            Vec3::new(9.0, -1.0, -half - 5.0),
            Vec3::new(11.0, 1.0, -half - 4.0),
        );
        let mut count = 0;
        hf.map_elements_in_local_aabb(&aabb, &mut |_id, _tri| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn project_local_point_onto_surface_returns_a_close_point() {
        let hf = flat(64, 16, 128); // ground at r≈15
        // A point well above the surface at (15, 0, 0)... should project to near (15, 0, 0)
        let pt = Vec3::new(20.0, 0.0, 0.0);
        let proj = hf.project_local_point(pt, false);
        // The projection should sit on the (locally piecewise-flat) surface, so its
        // x-coordinate is ≈ 15 (within the cell's triangulated approximation).
        assert!(
            proj.point.x > 14.0 && proj.point.x < 16.0,
            "unexpected projection x: {}",
            proj.point.x
        );
        // The projected point should be closer to the surface than the original.
        assert!((pt - proj.point).length() < (pt.length() - 14.0).abs() + 1.0);
    }
}
