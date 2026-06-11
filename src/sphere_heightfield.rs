//! A spherical heightmap collider — same heightmap data the renderer reads,
//! wrapped onto a sphere via Lambert equal-area cylindrical projection.
//!
//! The heightmap is parameterised by (u, v) where u spans the equator
//! (longitude θ = 2π · u / width, wrapping at width) and v spans the meridian
//! via v = (sin φ + 1) / 2 (latitude φ ∈ [-π/2, π/2]). For each sample, the
//! surface sits at ground_radius = lerp(radius_start, radius_end, height(u, v)),
//! measured from the origin in 3D — the same formula the rendering shader uses
//! (shaders/terrain-draw.wgsl, sphere branch).
//!
//! Bilinear interpolation is used between samples to produce a C0-smooth
//! surface, same trick as [`super::CylindricalHeightField`].
//!
//! A companion contact path inside [`super::CylDispatcher`] handles
//! `SphericalHeightField`-vs-Ball contacts directly against the smooth
//! surface — no triangle generation. Other shape-vs-heightfield pairs remain
//! unsupported (only wheels collide with terrain in this prototype).

use rapier3d::math::{Pose, Real, Vec3, Vector};
use rapier3d::parry::bounding_volume::{Aabb, BoundingSphere};
use rapier3d::parry::mass_properties::MassProperties;
use rapier3d::parry::query::{PointProjection, PointQuery, Ray, RayCast, RayIntersection};
use rapier3d::parry::shape::{Ball, FeatureId, Shape, ShapeType, TypedShape};
use std::sync::Arc;

#[derive(Clone)]
pub struct SphericalHeightField {
    /// Per-vertex heights packed as u8 (0..255 → 0.0..1.0). heights[v * width + u].
    /// Stored in an Arc so clone_dyn is cheap.
    heights: Arc<[u8]>,
    /// Number of samples around the equator. Wraps modulo this.
    width: u32,
    /// Number of samples along the meridian. v = 0 is the south pole row,
    /// v = height - 1 the north pole row.
    height: u32,
    radius_start: Real,
    radius_end: Real,
    aabb: Aabb,
}

impl SphericalHeightField {
    pub fn new(
        heights: Vec<u8>,
        width: u32,
        height: u32,
        radius_start: Real,
        radius_end: Real,
    ) -> Self {
        assert_eq!(
            heights.len() as u32,
            width * height,
            "heights size mismatch"
        );
        let r = radius_end;
        let aabb = Aabb::new(Vec3::new(-r, -r, -r), Vec3::new(r, r, r));
        Self {
            heights: heights.into(),
            width,
            height,
            radius_start,
            radius_end,
            aabb,
        }
    }

    pub fn radius_start(&self) -> Real {
        self.radius_start
    }
    pub fn radius_end(&self) -> Real {
        self.radius_end
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

    /// Bilinear sample of the height field at continuous (u, v) coordinates,
    /// returning (h, ∂h/∂u, ∂h/∂v). u wraps modulo width; v is clamped at the
    /// endpoints — the heightmap simply has no data outside [0, height-1].
    fn bilinear_h_with_grad(&self, u_cont: Real, v_cont: Real) -> (Real, Real, Real) {
        let cu = u_cont.floor() as i32;
        let mut cv = v_cont.floor() as i32;
        let u_frac = u_cont - cu as Real;
        let mut v_frac = v_cont - cv as Real;
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

    /// Smooth surface sample at world (theta, sin_phi). Returns:
    /// - `ground_radius`: distance from the origin to the surface at this (θ, φ).
    /// - `outward_normal`: unit normal pointing AWAY from the origin (into the
    ///   "sky" half-space where dynamic bodies live), tilted by the bilinear
    ///   surface gradient.
    ///
    /// theta wraps modulo 2π; sin_phi is clamped to [-1, 1].
    pub fn sample_surface(&self, theta: Real, sin_phi: Real) -> (Real, Vec3) {
        let two_pi = std::f32::consts::TAU;
        let theta_n = theta.rem_euclid(two_pi);
        let u_cont = (theta_n / two_pi) * self.width as Real;
        let s = sin_phi.clamp(-1.0, 1.0);
        // Lambert v = (sin φ + 1) / 2, scaled into pixel coordinates.
        let v_cont = ((s + 1.0) * 0.5) * (self.height - 1) as Real;

        let (h, dh_du, dh_dv) = self.bilinear_h_with_grad(u_cont, v_cont);
        let dr_dh = self.radius_end - self.radius_start;
        let ground_r = self.radius_start + h * dr_dh;

        // Surface point P(θ, s) = r · (c cos θ, c sin θ, s) where c = √(1-s²).
        // Tangents:
        //   ∂P/∂u = ∂r/∂u · radial + r · (-c sin θ, c cos θ, 0) · ∂θ/∂u
        //   ∂P/∂v = ∂r/∂v · radial + r · (-s/c cos θ, -s/c sin θ, 1) · ∂s/∂v
        // where ∂θ/∂u = 2π/width and ∂s/∂v = 2/(height-1) — Lambert maps the
        // pixel grid linearly in sin φ.
        let dr_du = dh_du * dr_dh;
        let dr_dv = dh_dv * dr_dh;
        let dtheta_du = two_pi / self.width as Real;
        let ds_dv = 2.0 / (self.height - 1) as Real;
        let cos_t = theta_n.cos();
        let sin_t = theta_n.sin();
        // Pad c away from 0 at the poles to keep the normal finite (the chassis
        // would still feel the heightmap's pole-row value as a flat top).
        let c = (1.0 - s * s).max(0.0).sqrt().max(1e-3);
        let radial = Vec3::new(c * cos_t, c * sin_t, s);
        let dp_du = dr_du * radial
            + Vec3::new(-ground_r * c * sin_t, ground_r * c * cos_t, 0.0) * dtheta_du;
        let dp_dv = dr_dv * radial
            + Vec3::new(
                -ground_r * s / c * cos_t,
                -ground_r * s / c * sin_t,
                ground_r,
            ) * ds_dv;
        let n_unnorm = dp_du.cross(dp_dv);
        let n_len_sq = n_unnorm.length_squared();
        let normal = if n_len_sq > 1e-12 {
            n_unnorm / n_len_sq.sqrt()
        } else {
            radial
        };
        (ground_r, normal)
    }
}

impl Shape for SphericalHeightField {
    fn compute_local_aabb(&self) -> Aabb {
        self.aabb
    }

    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        BoundingSphere::new(Vec3::ZERO, self.radius_end)
    }

    fn clone_dyn(&self) -> Box<dyn Shape> {
        Box::new(self.clone())
    }

    fn scale_dyn(&self, _scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        None
    }

    fn mass_properties(&self, density: Real) -> MassProperties {
        // Same trick as the cylindrical version: pretend we're a solid ball of
        // average radius so the radial-gravity formula (which reads
        // `terrain.mass()`) still sees a sensible mass.
        let r = 0.5 * (self.radius_start + self.radius_end);
        Ball::new(r).mass_properties(density)
    }

    fn shape_type(&self) -> ShapeType {
        ShapeType::Custom
    }

    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(self)
    }

    fn ccd_thickness(&self) -> Real {
        // Smallest cell dimension at the equator (worst case along both axes).
        let cu = std::f32::consts::TAU * self.radius_start / self.width as Real;
        let cv = std::f32::consts::PI * self.radius_start / (self.height - 1) as Real;
        0.5 * cu.min(cv)
    }

    fn ccd_angular_thickness(&self) -> Real {
        std::f32::consts::FRAC_PI_4
    }
}

impl PointQuery for SphericalHeightField {
    fn project_local_point(&self, pt: Vec3, _solid: bool) -> PointProjection {
        // Radial projection of the query onto the surface. Same approximation
        // the cylindrical version uses — good enough for the few rapier queries
        // that hit this shape.
        let r_pt = pt.length();
        if r_pt < 1e-6 {
            return PointProjection::new(true, Vec3::new(self.radius_start, 0.0, 0.0));
        }
        let unit = pt / r_pt;
        let theta = pt.y.atan2(pt.x);
        let (ground_r, _normal) = self.sample_surface(theta, unit.z);
        let surface_pt = unit * ground_r;
        PointProjection::new(r_pt < ground_r, surface_pt)
    }

    fn project_local_point_and_get_feature(&self, pt: Vec3) -> (PointProjection, FeatureId) {
        (self.project_local_point(pt, true), FeatureId::Unknown)
    }
}

impl RayCast for SphericalHeightField {
    fn cast_local_ray_and_get_normal(
        &self,
        _ray: &Ray,
        _max_time_of_impact: Real,
        _solid: bool,
    ) -> Option<RayIntersection> {
        // Stub — same as the cylindrical version; no scene-ray queries hit the
        // terrain at the moment.
        None
    }
}

/// Sphere-vs-ball contact, used by [`super::CylDispatcher`]. Builds a single
/// manifold against the smooth surface at the ball centre's (θ, φ).
pub(super) fn sphere_vs_ball<ManifoldData, ContactData>(
    pos12: &Pose,
    hf: &SphericalHeightField,
    ball: &Ball,
    prediction: Real,
    manifolds: &mut Vec<rapier3d::parry::query::ContactManifold<ManifoldData, ContactData>>,
    flipped: bool,
) where
    ManifoldData: Default + Clone,
    ContactData: Default + Copy,
{
    use rapier3d::parry::query::{ContactManifold, TrackedContact};
    use rapier3d::parry::shape::PackedFeatureId;
    manifolds.clear();
    let c = pos12.translation;
    let r_c = c.length();
    if r_c < 1e-6 {
        return; // Ball sitting on the sphere centre is degenerate; skip.
    }
    let unit = c / r_c;
    let theta = c.y.atan2(c.x);
    let (ground_r, normal_hf) = hf.sample_surface(theta, unit.z);
    let surface_pt = unit * ground_r;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: u32, height: u32, alpha: u8) -> SphericalHeightField {
        SphericalHeightField::new(
            vec![alpha; (width * height) as usize],
            width,
            height,
            10.0,
            20.0,
        )
    }

    #[test]
    fn sample_surface_on_flat_is_radial() {
        let hf = flat(16, 8, 0); // alpha = 0 → ground at radius_start = 10.
                                 // Equator point.
        let (r, n) = hf.sample_surface(0.0, 0.0);
        assert!((r - 10.0).abs() < 1e-4);
        assert!((n - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3);
        // Near the pole.
        let (r2, n2) = hf.sample_surface(0.0, 0.95);
        assert!((r2 - 10.0).abs() < 1e-4);
        // The outward normal should still point ~radially outward at the pole.
        let expected = Vec3::new((1.0_f32 - 0.95 * 0.95).max(0.0).sqrt(), 0.0, 0.95);
        let n2u = n2 / n2.length();
        let eu = expected / expected.length();
        assert!((n2u - eu).length() < 5e-2);
    }

    #[test]
    fn sample_surface_height_increases_with_alpha() {
        let hf = flat(16, 8, 255); // alpha = 1 → ground at radius_end = 20.
        let (r, _) = hf.sample_surface(0.0, 0.0);
        assert!((r - 20.0).abs() < 1e-4);
    }

    #[test]
    fn boozeena_average_ground_matches_heightmap_stats() {
        use std::io::BufReader;
        let map = std::path::Path::new("data/maps/boozeena/map.png");
        let Ok(file) = std::fs::File::open(map) else {
            // Allow the test to be skipped when the map isn't checked out
            // (git-lfs not pulled). The point is to validate parity, not
            // gate CI on the heightmap file.
            return;
        };
        let decoder = png::Decoder::new(BufReader::new(file));
        let mut reader = decoder.read_info().expect("png header");
        let size = reader.output_buffer_size().expect("png size");
        let mut decoded = vec![0u8; size];
        let info = reader.next_frame(&mut decoded).expect("png decode");
        let alpha: Vec<u8> = (0..info.width as usize * info.height as usize)
            .map(|i| decoded[i * 4 + 3])
            .collect();
        let hf = SphericalHeightField::new(alpha.clone(), info.width, info.height, 10.0, 20.0);
        // Average ground_r over 100 random (theta, sin_phi) points should match
        // the Python-side analysis (~11.87). Anything substantially higher
        // means sample_surface is reading the wrong texels or misinterpreting
        // the alpha → radius mapping.
        let mut rng_state: u64 = 0xdead_beef_cafe_babe;
        let mut sum = 0.0;
        let n = 100;
        for _ in 0..n {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 32) as u32 as f32 / (u32::MAX as f32 + 1.0);
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (rng_state >> 32) as u32 as f32 / (u32::MAX as f32 + 1.0);
            let theta = u * std::f32::consts::TAU;
            let sin_phi = v * 2.0 - 1.0;
            let (r, _) = hf.sample_surface(theta, sin_phi);
            sum += r;
        }
        let avg = sum / n as f32;
        eprintln!("boozeena physics-surface avg ground_r over {n} samples: {avg:.3}");
        assert!(
            (avg - 11.87).abs() < 0.5,
            "physics-side ground_r avg {avg:.3} drifts from heightmap avg 11.87"
        );
    }
}
