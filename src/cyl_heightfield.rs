//! A cylindrical heightmap collider — full-resolution heightfield wrapped around the Z axis.
//!
//! The heightmap is parameterized by (u, v) where u spans the circumference (theta = 2π·u/width,
//! wrapping at width) and v spans the cylinder's axis (z = -L/2 + v·L/(height-1)). For each
//! sample, the surface lies at ground_radius = lerp(radius_start, radius_end, height(u, v)) —
//! the same formula the rendering shader uses (shaders/terrain-draw.wgsl).
//!
//! Bilinear interpolation is used between samples to produce a C0-smooth surface
//! (continuous values everywhere; gradient continuous within each cell, with kinks at cell
//! boundaries). This avoids the wheel-jitter that triangulated heightfields produce when
//! wheels cross the diagonal of every 3 cm cell.
//!
//! A companion [`CylDispatcher`] plugs into rapier's NarrowPhase to handle
//! `CylindricalHeightField`-vs-Ball contact directly against the smooth surface — no
//! triangle generation. Other shape-vs-heightfield pairs are unsupported (in this
//! prototype only wheels collide with terrain).

use rapier3d::math::{Pose, Real, Vec3, Vector};
use rapier3d::parry::bounding_volume::{Aabb, BoundingSphere};
use rapier3d::parry::mass_properties::MassProperties;
use rapier3d::parry::query::details::NormalConstraints;
use rapier3d::parry::query::{
    ClosestPoints, Contact, ContactManifold, ContactManifoldsWorkspace, DefaultQueryDispatcher,
    NonlinearRigidMotion, PersistentQueryDispatcher, PointProjection, PointQuery, QueryDispatcher,
    Ray, RayCast, RayIntersection, ShapeCastHit, ShapeCastOptions, TrackedContact, Unsupported,
};
use rapier3d::parry::shape::{
    Ball, Cylinder, FeatureId, PackedFeatureId, Shape, ShapeType, TypedShape,
};
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
        assert_eq!(
            heights.len() as u32,
            width * height,
            "heights size mismatch"
        );
        let r = radius_end;
        let half_len = 0.5 * length;
        let aabb = Aabb::new(Vec3::new(-r, -r, -half_len), Vec3::new(r, r, half_len));
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
    /// at grid coords (u, v). u wraps via modulo; v is clamped. Kept for tests/visualisation;
    /// the runtime contact path uses [`Self::sample_surface`] (bilinear) instead.
    #[inline]
    pub fn vertex(&self, u: i32, v: i32) -> Vec3 {
        // Use the *unwrapped* u for theta so neighbour vertices stay angularly adjacent
        // (otherwise a cell straddling u=width-1→u=0 would degenerate).
        let theta = (u as Real / self.width as Real) * std::f32::consts::TAU;
        let v_clamped = v.clamp(0, self.height as i32 - 1);
        let z = -0.5 * self.length + (v_clamped as Real / (self.height - 1) as Real) * self.length;
        let h = self.h(u, v);
        let r = self.radius_start + h * (self.radius_end - self.radius_start);
        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }

    /// Bilinear sample of the height field at continuous (u, v) coordinates with gradient.
    /// Returns (h, ∂h/∂u, ∂h/∂v) where h ∈ [0,1] is the alpha-normalized height.
    /// u wraps modulo `width`; v is clamped at endpoints (no extrapolation off the ends).
    fn bilinear_h_with_grad(&self, u_cont: Real, v_cont: Real) -> (Real, Real, Real) {
        let cu = u_cont.floor() as i32;
        let mut cv = v_cont.floor() as i32;
        let u_frac = u_cont - cu as Real;
        let mut v_frac = v_cont - cv as Real;
        // Clamp v at endpoints — no extrapolation beyond the heightfield's axial extent.
        let v_max = self.height as i32 - 2;
        if cv < 0 {
            cv = 0;
            v_frac = 0.0;
        }
        if cv > v_max {
            cv = v_max;
            v_frac = 1.0;
        }
        let h00 = self.h(cu, cv);
        let h10 = self.h(cu + 1, cv);
        let h01 = self.h(cu, cv + 1);
        let h11 = self.h(cu + 1, cv + 1);
        let h = (1.0 - u_frac) * (1.0 - v_frac) * h00
            + u_frac * (1.0 - v_frac) * h10
            + (1.0 - u_frac) * v_frac * h01
            + u_frac * v_frac * h11;
        let dh_du = (1.0 - v_frac) * (h10 - h00) + v_frac * (h11 - h01);
        let dh_dv = (1.0 - u_frac) * (h01 - h00) + u_frac * (h11 - h10);
        (h, dh_du, dh_dv)
    }

    /// Smooth surface sample at world (theta, z). Returns:
    /// - `ground_radius`: the radial distance from the cylinder axis to the surface at this
    ///   (theta, z).
    /// - `outward_normal`: unit normal pointing AWAY from the cylinder axis (i.e. into the
    ///   region where dynamic bodies live), tilted by the bilinear surface gradient.
    ///
    /// theta wraps modulo 2π; z is clamped to the axial extent.
    pub fn sample_surface(&self, theta: Real, z: Real) -> (Real, Vec3) {
        let two_pi = std::f32::consts::TAU;
        let theta_n = theta.rem_euclid(two_pi);
        let u_cont = (theta_n / two_pi) * self.width as Real;
        let half_len = 0.5 * self.length;
        let z_clamped = z.clamp(-half_len, half_len);
        let v_cont = ((z_clamped + half_len) / self.length) * (self.height - 1) as Real;

        let (h, dh_du, dh_dv) = self.bilinear_h_with_grad(u_cont, v_cont);
        let dr_dh = self.radius_end - self.radius_start;
        let ground_r = self.radius_start + h * dr_dh;

        // Surface point P(theta, z) = (r cos θ, r sin θ, z). Tangents along the two
        // parameter directions:
        //   ∂P/∂u = ((∂r/∂u) cos θ − r sin θ · ∂θ/∂u,
        //           (∂r/∂u) sin θ + r cos θ · ∂θ/∂u,
        //           0)
        //   ∂P/∂v = ((∂r/∂v) cos θ, (∂r/∂v) sin θ, ∂z/∂v)
        // where ∂r/∂u = (∂h/∂u)·(r_end − r_start), ∂r/∂v likewise,
        //       ∂θ/∂u = 2π/width, ∂z/∂v = L/(height−1).
        let dr_du = dh_du * dr_dh;
        let dr_dv = dh_dv * dr_dh;
        let dtheta_du = two_pi / self.width as Real;
        let dz_dv = self.length / (self.height - 1) as Real;
        let cos_t = theta_n.cos();
        let sin_t = theta_n.sin();
        let dp_du = Vec3::new(
            dr_du * cos_t - ground_r * sin_t * dtheta_du,
            dr_du * sin_t + ground_r * cos_t * dtheta_du,
            0.0,
        );
        let dp_dv = Vec3::new(dr_dv * cos_t, dr_dv * sin_t, dz_dv);
        // dp_du × dp_dv evaluates to (outward radial) × (axial+radial) which for a flat
        // surface gives a vector in +radial direction (verified: for dr_du=dr_dv=0, cross is
        // r·dθ/du·dz/dv · (cos θ, sin θ, 0) — outward radial).
        let n_unnorm = dp_du.cross(dp_dv);
        let n_len_sq = n_unnorm.length_squared();
        let normal = if n_len_sq > 1e-12 {
            n_unnorm / n_len_sq.sqrt()
        } else {
            Vec3::new(cos_t, sin_t, 0.0)
        };

        (ground_r, normal)
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
        // Closest point on the smooth surface approximated by projecting the query point
        // radially onto the surface at its (θ, z). For points not exactly above the
        // closest surface patch this isn't the *true* closest point — a Newton refinement
        // would do that — but it's good enough for the few rapier queries that hit this.
        let theta = pt.y.atan2(pt.x);
        let (ground_r, normal) = self.sample_surface(theta, pt.z);
        let surface_pt = Vec3::new(ground_r * theta.cos(), ground_r * theta.sin(), pt.z);
        let r_pt = (pt.x * pt.x + pt.y * pt.y).sqrt();
        let is_inside = r_pt < ground_r;
        let _ = normal;
        PointProjection::new(is_inside, surface_pt)
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

/// Narrow-phase dispatcher that handles `CylindricalHeightField`-vs-Ball contacts directly
/// against the smooth bilinear surface (no triangulation, no triangle BVH). For other shapes
/// against the heightfield, this returns no contacts — in this prototype only wheels (balls)
/// touch the terrain (chassis colliders set their collision group to none()).
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

    /// Build a single contact manifold for `ball` against `hf`. Soft-tire
    /// approximation: average ground height + surface normal across the
    /// wheel's footprint (5 samples in a `+` pattern) so the wheel "feels" a
    /// locally-smoothed surface instead of the exact bilinear sample under
    /// its centre. The visual heightmap is unchanged; this only changes what
    /// the wheel-vs-terrain collision sees. Reduces micro-bouncing and the
    /// "stuck on tiny ridge" behaviour without going to true multi-point
    /// contact (which destabilises rapier's PGS solver in our setup).
    fn cyl_vs_ball<ManifoldData, ContactData>(
        &self,
        pos12: &Pose,
        hf: &CylindricalHeightField,
        ball: &Ball,
        prediction: Real,
        manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
        flipped: bool,
    ) where
        ManifoldData: Default + Clone,
        ContactData: Default + Copy,
    {
        manifolds.clear();
        let c = pos12.translation;
        let theta = c.y.atan2(c.x);

        // Sample the footprint and average. Larger ratio = softer tire.
        const SOFT_FOOTPRINT_RATIO: Real = 0.5;
        let r_off = ball.radius * SOFT_FOOTPRINT_RATIO;
        let r_avg = 0.5 * (hf.radius_start + hf.radius_end);
        let dtheta = r_off / r_avg;
        let samples: [(Real, Real); 5] = [
            (0.0, 0.0),
            (dtheta, 0.0),
            (-dtheta, 0.0),
            (0.0, r_off),
            (0.0, -r_off),
        ];
        let mut sum_gr = 0.0;
        let mut sum_n = Vec3::ZERO;
        for (dt, dz) in samples {
            let (gr, nh) = hf.sample_surface(theta + dt, c.z + dz);
            sum_gr += gr;
            sum_n += nh;
        }
        let n_samples = samples.len() as Real;
        let ground_r = sum_gr / n_samples;
        let n_len = sum_n.length();
        let normal_hf = if n_len > 1e-6 {
            sum_n / n_len
        } else {
            Vec3::new(theta.cos(), theta.sin(), 0.0)
        };

        let surface_pt = Vec3::new(ground_r * theta.cos(), ground_r * theta.sin(), c.z);
        let signed_center_dist = (c - surface_pt).dot(normal_hf);
        let dist = signed_center_dist - ball.radius;
        if dist >= prediction {
            return;
        }

        let (local_n1, local_n2, local_p1, local_p2) = if flipped {
            let n_ball = pos12.rotation.inverse() * (-normal_hf);
            let p_ball_in_ball = n_ball * ball.radius;
            (n_ball, normal_hf, p_ball_in_ball, surface_pt)
        } else {
            let n_ball = pos12.rotation.inverse() * (-normal_hf);
            let p_ball_in_ball = n_ball * ball.radius;
            (normal_hf, n_ball, surface_pt, p_ball_in_ball)
        };
        let mut manifold =
            ContactManifold::<ManifoldData, ContactData>::with_data(0, 0, ManifoldData::default());
        manifold.local_n1 = local_n1;
        manifold.local_n2 = local_n2;
        let fid = PackedFeatureId::face(0);
        manifold
            .points
            .push(TrackedContact::new(local_p1, local_p2, fid, fid, dist));
        manifolds.push(manifold);
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

    fn distance(&self, pos12: &Pose, g1: &dyn Shape, g2: &dyn Shape) -> Result<Real, Unsupported> {
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
        self.inner.cast_shapes(pos12, vel12, g1, g2, options)
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

impl<ManifoldData, ContactData> PersistentQueryDispatcher<ManifoldData, ContactData>
    for CylDispatcher
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
            if let Some(ball) = g2.as_ball() {
                self.cyl_vs_ball::<ManifoldData, ContactData>(
                    pos12, hf, ball, prediction, manifolds, false,
                );
            } else {
                manifolds.clear();
            }
            return Ok(());
        }
        if let Some(hf) = Self::try_extract(g2) {
            if let Some(ball) = g1.as_ball() {
                self.cyl_vs_ball::<ManifoldData, ContactData>(
                    &pos12.inverse(),
                    hf,
                    ball,
                    prediction,
                    manifolds,
                    true,
                );
            } else {
                manifolds.clear();
            }
            return Ok(());
        }
        // Spherical heightfield branch — same routing pattern as the cylinder.
        if let Some(sh) = g1.downcast_ref::<super::SphericalHeightField>() {
            if let Some(ball) = g2.as_ball() {
                super::sphere_heightfield::sphere_vs_ball::<ManifoldData, ContactData>(
                    pos12, sh, ball, prediction, manifolds, false,
                );
            } else {
                manifolds.clear();
            }
            return Ok(());
        }
        if let Some(sh) = g2.downcast_ref::<super::SphericalHeightField>() {
            if let Some(ball) = g1.as_ball() {
                super::sphere_heightfield::sphere_vs_ball::<ManifoldData, ContactData>(
                    &pos12.inverse(),
                    sh,
                    ball,
                    prediction,
                    manifolds,
                    true,
                );
            } else {
                manifolds.clear();
            }
            return Ok(());
        }
        self.inner
            .contact_manifolds(pos12, g1, g2, prediction, manifolds, workspace)
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
        let idx = |u: u32, v: u32| (v * width + u) as usize;
        heights[idx(0, 1)] = 255; // (u=0, v=1) → alpha=1 → r=20
        heights[idx(0, 2)] = 128; // (u=0, v=2) → alpha≈0.5 → r≈15
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
    fn sample_surface_on_flat_returns_radial_normal() {
        let hf = flat(64, 32, 128); // ground at r ≈ 15.0196
        let (ground_r, normal) = hf.sample_surface(0.0, 0.0);
        assert!(
            (ground_r - 15.0196).abs() < 0.01,
            "got ground_r = {ground_r}"
        );
        // For a flat cylinder (uniform alpha), the outward normal points radially.
        assert!(
            approx_eq(normal, Vec3::new(1.0, 0.0, 0.0), 1e-3),
            "{:?}",
            normal
        );

        let (_, n_at_pi_2) = hf.sample_surface(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(
            approx_eq(n_at_pi_2, Vec3::new(0.0, 1.0, 0.0), 1e-3),
            "{:?}",
            n_at_pi_2
        );
    }

    #[test]
    fn sample_surface_interpolates_height_between_samples() {
        // Two adjacent samples with very different alpha → midpoint should land halfway.
        let width = 4;
        let height = 4;
        let mut heights = vec![0u8; (width * height) as usize];
        let idx = |u: u32, v: u32| (v * width + u) as usize;
        heights[idx(0, 1)] = 0; // alpha=0 → r=10
        heights[idx(1, 1)] = 255; // alpha=1 → r=20
        heights[idx(0, 2)] = 0;
        heights[idx(1, 2)] = 255;
        let hf = CylindricalHeightField::new(heights, width, height, 10.0, 20.0, 30.0);

        let theta_at_u_0 = 0.0;
        let theta_at_u_1 = std::f32::consts::TAU / width as f32;
        let theta_mid = 0.5 * (theta_at_u_0 + theta_at_u_1);
        let z_mid_v_1_2 = -15.0 + (1.5 / (height - 1) as f32) * 30.0;

        let (r_mid, _) = hf.sample_surface(theta_mid, z_mid_v_1_2);
        // u_frac = 0.5, v_frac = 0.5, h = bilerp(0,1,0,1) = 0.5 → r = 15
        assert!((r_mid - 15.0).abs() < 0.01, "got r_mid = {r_mid}");
    }

    #[test]
    fn sample_surface_normal_tilts_on_slope() {
        // Build a tiny heightfield with a slope along v (the z-axis direction).
        let width = 4;
        let height = 4;
        let mut heights = vec![0u8; (width * height) as usize];
        for v in 0..height {
            for u in 0..width {
                // Alpha increases linearly with v (and is uniform in u): a ramp in z.
                heights[(v * width + u) as usize] =
                    ((v as f32 / (height - 1) as f32) * 255.0) as u8;
            }
        }
        let hf = CylindricalHeightField::new(heights, width, height, 10.0, 20.0, 30.0);

        // Sample somewhere in the middle.
        let (_, normal) = hf.sample_surface(0.0, 0.0);
        // The outward normal should have a tilt in the +radial (+x) direction with a
        // -z component (slope rises with z, so outward normal leans toward -z).
        assert!(
            normal.x > 0.0,
            "expected outward (radial) component, got {:?}",
            normal
        );
        assert!(
            normal.z.abs() > 0.01,
            "expected slope-induced z tilt, got {:?}",
            normal
        );
        // Unit length.
        assert!((normal.length() - 1.0).abs() < 1e-3, "{:?}", normal);
    }

    #[test]
    fn project_local_point_onto_surface_returns_a_close_point() {
        let hf = flat(64, 16, 128); // ground at r≈15
        let pt = Vec3::new(20.0, 0.0, 0.0);
        let proj = hf.project_local_point(pt, false);
        assert!(
            proj.point.x > 14.5 && proj.point.x < 15.5,
            "unexpected projection x: {}",
            proj.point.x
        );
    }
}
