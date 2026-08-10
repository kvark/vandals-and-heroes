//! Triangulated irregular network (TIN) approximation of the terrain.
//!
//! Instead of ray-marching or voxel-casting the height map, this builds an
//! actual triangle mesh that approximates it, following Garland & Heckbert's
//! greedy insertion scheme ("Fast Polygonal Approximation of Terrains and
//! Height Fields", 1995): start from a coarse triangulation, then repeatedly
//! insert the grid point with the largest vertical error until the whole
//! surface is within a target tolerance.
//!
//! The triangulation runs on the flat (u, v) texel grid of the height map;
//! only at emit time are vertices mapped onto the world surface — a
//! cylinder, a sphere, or a torus (see [`Mapping::embed`]). The same mesh
//! feeds both the renderer (per-chunk vertex/index buffers with LODs) and
//! the physics (per-chunk trimesh colliders at LOD 0), so what you see is
//! exactly what you collide with.
//!
//! The triangulation uses chunk-local integer coordinates so that the
//! `orient2d` / `in_circle` predicates are exact in `i64` — the grid is
//! massively cocircular (every axis-aligned square of four samples), which
//! is exactly the case floating-point predicates get wrong.

use crate::config::{Map as MapConfig, WorldShape};

/// Sentinel for "no triangle" in adjacency links and slot indices.
const NONE: u32 = u32::MAX;

/// Side of a build chunk, in texels. Large enough that the border
/// simplification stays a small fraction of the vertex budget, small enough
/// to keep the greedy rasterisation cheap, give the worker threads plenty of
/// independent work, and keep per-chunk GPU buffers far below WebGL-class
/// buffer limits.
const CHUNK_SIZE: u32 = 128;

/// Discrete level-of-detail steps kept per chunk. Each step doubles the fit
/// tolerance, which roughly halves the triangle count.
pub const LOD_COUNT: usize = 3;

/// World-space mapping of the (u, v) height-map grid.
///
/// The conventions mirror `shaders/terrain-mesh.wgsl` exactly — sample `i`
/// of a row is the texel centre at `u = (i + 0.5) / width`, and:
///
/// * **Cylinder**: `θ = u·2π`, `z = (v − 0.5)·length`, radius from the
///   height byte. `u` wraps, `v` clamps.
/// * **Sphere**: Lambert equal-area cylindrical projection —
///   `θ = u·2π`, `sin φ = 2v − 1`. `u` wraps, `v` clamps (the pole rows
///   collapse toward points).
/// * **Torus**: the cylinder's axis bent into a circle of major radius
///   `R = length / 2π` in the XY plane: `θ` becomes the tube angle and the
///   axial coordinate becomes the major angle `φ = (v − 0.5)·2π`. Both `u`
///   and `v` wrap — the world has no ends.
#[derive(Clone, Copy)]
pub struct Mapping {
    pub shape: WorldShape,
    pub radius_start: f32,
    pub radius_end: f32,
    pub length: f32,
    pub width: u32,
    pub height: u32,
}

impl Mapping {
    pub fn new(config: &MapConfig, width: u32, height: u32) -> Self {
        Self {
            shape: config.shape,
            radius_start: config.radius.start,
            radius_end: config.radius.end,
            length: config.length,
            width,
            height,
        }
    }

    /// Major radius of the torus centreline; meaningless for other shapes.
    pub fn major_radius(&self) -> f32 {
        self.length / std::f32::consts::TAU
    }

    pub fn ground_radius(&self, h01: f32) -> f32 {
        self.radius_start + h01 * (self.radius_end - self.radius_start)
    }

    /// Texel-space `(x, y)` (continuous, texel centre at integer + 0.5) and
    /// a raw height byte to a world position.
    pub fn embed(&self, x: f32, y: f32, h_byte: f32) -> [f32; 3] {
        use std::f32::consts::TAU;
        let u = x / self.width as f32;
        let v = y / self.height as f32;
        let r = self.ground_radius(h_byte / 255.0);
        let theta = u * TAU;
        let (sin_t, cos_t) = theta.sin_cos();
        match self.shape {
            WorldShape::Cylinder => {
                let z = (v - 0.5) * self.length;
                [r * cos_t, r * sin_t, z]
            }
            WorldShape::Sphere => {
                let s = (2.0 * v - 1.0).clamp(-1.0, 1.0); // sin φ
                let c = (1.0 - s * s).max(0.0).sqrt(); // cos φ
                [r * c * cos_t, r * c * sin_t, r * s]
            }
            WorldShape::Torus => {
                let big_r = self.major_radius();
                let phi = (v - 0.5) * TAU;
                let (sin_p, cos_p) = phi.sin_cos();
                let ring = big_r + r * cos_t;
                [ring * cos_p, ring * sin_p, r * sin_t]
            }
        }
    }

    /// Whether the `v` (axial) direction wraps around.
    fn wrap_y(&self) -> bool {
        matches!(self.shape, WorldShape::Torus)
    }

    /// With `(u, v)`-CCW triangles, the world-space winding comes out
    /// inward-facing on the torus (the bend flips handedness); flip the
    /// emitted triangles there so every shape ends up with outward-CCW
    /// faces.
    fn flip_winding(&self) -> bool {
        matches!(self.shape, WorldShape::Torus)
    }

    /// Radial tolerance in world units for a tolerance in height bytes.
    fn tol_world(&self, tol_bytes: f32) -> f32 {
        tol_bytes / 255.0 * (self.radius_end - self.radius_start)
    }

    /// Maximum vertex spacing (texels) along u and v so that a straight
    /// chord between neighbouring vertices deviates from the *curved* world
    /// surface by at most `tol_world`, over the chunk covering texel rows
    /// `[y0 ..= y0 + h]`.
    ///
    /// The greedy fit only measures height error in (u, v) space; a flat
    /// height region fits with two giant triangles there, but its world
    /// image is an arc, and a chord across arc `a` on curvature radius `ρ`
    /// sags by `ρ·(1 − cos(a/2)) ≤ ρ·a²/8`. Bounding that by the tolerance
    /// gives `a ≤ sqrt(8·tol/ρ)`. `None` = the direction is straight.
    ///
    /// Two subtleties:
    ///
    /// * On shapes where *both* directions curve (sphere, torus), a lattice
    ///   cell's diagonal accumulates the sagitta of both, so each direction
    ///   only gets half the budget.
    /// * The sphere's Lambert parameterisation packs more latitude per texel
    ///   the closer to a pole the row sits (`dφ/dv = 2/cos φ`), so the `v`
    ///   step comes from the chunk's own worst row. Pole-adjacent chunks
    ///   densify; the seams stay crack-free because horizontally adjacent
    ///   chunks share their row range and therefore derive the same step.
    fn curvature_steps(&self, tol_world: f32, y0: i32, h: u32) -> (Option<u32>, Option<u32>) {
        use std::f32::consts::TAU;
        let split = match self.shape {
            WorldShape::Cylinder => 1.0,
            WorldShape::Sphere | WorldShape::Torus => 0.5,
        };
        let tol = (tol_world * split).max(1e-4);
        let arc_for = |rho: f32| (8.0 * tol / rho).sqrt();
        let step = |arc: f32, angle_per_texel: f32| -> Option<u32> {
            Some(((arc / angle_per_texel) as u32).max(1))
        };
        let r = self.radius_end;
        let u_step = step(arc_for(r), TAU / self.width as f32);
        let v_step = match self.shape {
            WorldShape::Cylinder => None,
            WorldShape::Sphere => {
                // Worst (pole-most) row of the chunk: |sin φ| is maximal at
                // the range ends. `dφ = (2/height)/cos φ · dv_texels`.
                let s_at = |y: f32| (2.0 * y / self.height as f32 - 1.0).clamp(-1.0, 1.0);
                let s_worst = s_at(y0 as f32)
                    .abs()
                    .max(s_at((y0 + h as i32 + 1) as f32).abs());
                let cos_phi = (1.0 - s_worst * s_worst).max(0.0).sqrt();
                let angle_per_texel = 2.0 / self.height as f32 / cos_phi.max(1e-4);
                step(arc_for(r), angle_per_texel)
            }
            WorldShape::Torus => {
                step(arc_for(self.major_radius() + r), TAU / self.height as f32)
            }
        };
        (u_step, v_step)
    }

    /// Height byte at a texel, applying the shape's wrap/clamp rules.
    fn sample(&self, alpha: &[u8], x: i32, y: i32) -> f32 {
        let x = x.rem_euclid(self.width as i32);
        let y = if self.wrap_y() {
            y.rem_euclid(self.height as i32)
        } else {
            y.clamp(0, self.height as i32 - 1)
        };
        alpha[y as usize * self.width as usize + x as usize] as f32
    }
}

/// The single quality knob, in `0..=1`, mapping to a vertical tolerance in
/// height-byte units. `1.0` asks for one byte — the finest detail the 8-bit
/// samples can carry. Every 0.25 below that doubles the tolerance, bottoming
/// out at 16 bytes.
pub fn max_error_for_quality(quality: f32) -> f32 {
    (4.0 * (1.0 - quality.clamp(0.0, 1.0))).exp2()
}

/// Twice the signed area of the triangle `abc`; positive when CCW.
/// Exact: chunk-local coordinates are bounded by `CHUNK_SIZE`.
fn orient2d(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i64 {
    let (ax, ay) = (a[0] as i64, a[1] as i64);
    let (bx, by) = (b[0] as i64, b[1] as i64);
    let (cx, cy) = (c[0] as i64, c[1] as i64);
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// Positive when `d` lies strictly inside the circumcircle of the CCW
/// triangle `abc`. Exact for the same reason as `orient2d`.
fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i64 {
    let adx = (a[0] - d[0]) as i64;
    let ady = (a[1] - d[1]) as i64;
    let bdx = (b[0] - d[0]) as i64;
    let bdy = (b[1] - d[1]) as i64;
    let cdx = (c[0] - d[0]) as i64;
    let cdy = (c[1] - d[1]) as i64;

    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    adx * (bdy * clift - cdy * blift) - ady * (bdx * clift - cdx * blift)
        + alift * (bdx * cdy - cdx * bdy)
}

#[derive(Clone)]
struct Tri {
    v: [u32; 3],
    /// `n[i]` is the triangle across edge `(v[i], v[(i + 1) % 3])`.
    n: [u32; 3],
    /// Grid index of the worst-approximated sample inside this triangle.
    cand: u32,
    err: f32,
    alive: bool,
}

/// One chunk's sample grid: `nx * ny` height samples covering texels
/// `[x0 ..= x0 + w] × [y0 ..= y0 + h]`, i.e. neighbouring chunks overlap by
/// exactly one row/column so their boundary vertices coincide.
struct Grid {
    heights: Vec<f32>,
    nx: u32,
    ny: u32,
    x0: i32,
    y0: i32,
}

impl Grid {
    fn new(mapping: &Mapping, alpha: &[u8], x0: i32, y0: i32, w: u32, h: u32) -> Self {
        let nx = w + 1;
        let ny = h + 1;
        let mut heights = Vec::with_capacity((nx * ny) as usize);
        for ly in 0..ny {
            for lx in 0..nx {
                heights.push(mapping.sample(alpha, x0 + lx as i32, y0 + ly as i32));
            }
        }
        Grid {
            heights,
            nx,
            ny,
            x0,
            y0,
        }
    }

    fn index(&self, lx: u32, ly: u32) -> u32 {
        ly * self.nx + lx
    }

    fn coord(&self, index: u32) -> [i32; 2] {
        [(index % self.nx) as i32, (index / self.nx) as i32]
    }

    fn height(&self, index: u32) -> f32 {
        self.heights[index as usize]
    }
}

/// Greedy TIN over a single chunk.
#[cfg_attr(test, derive(Clone))]
struct Chunk {
    /// Grid index of each triangulation vertex.
    verts: Vec<u32>,
    tris: Vec<Tri>,
    /// Freed slots from collapsed cavities, reused before growing `tris`.
    free: Vec<u32>,
}

impl Chunk {
    fn pos(&self, grid: &Grid, v: u32) -> [i32; 2] {
        grid.coord(self.verts[v as usize])
    }

    fn height(&self, grid: &Grid, v: u32) -> f32 {
        grid.height(self.verts[v as usize])
    }

    /// Seed with the two triangles spanning the chunk rectangle.
    fn new(grid: &Grid) -> Self {
        let (mx, my) = (grid.nx - 1, grid.ny - 1);
        let corners = [
            grid.index(0, 0),
            grid.index(mx, 0),
            grid.index(mx, my),
            grid.index(0, my),
        ];
        let tris = vec![
            Tri {
                v: [0, 1, 2],
                n: [NONE, NONE, 1],
                cand: NONE,
                err: 0.0,
                alive: true,
            },
            Tri {
                v: [0, 2, 3],
                n: [0, NONE, NONE],
                cand: NONE,
                err: 0.0,
                alive: true,
            },
        ];
        Chunk {
            verts: corners.to_vec(),
            tris,
            free: Vec::new(),
        }
    }

    /// Walk from `hint` toward `p` until we land in the triangle containing
    /// it. The domain stays convex (it is always the chunk rectangle), so a
    /// straight walk terminates; the cap is pure paranoia.
    fn locate(&self, grid: &Grid, p: [i32; 2], hint: u32) -> u32 {
        let mut t = hint;
        for _ in 0..self.tris.len() + 8 {
            let tri = &self.tris[t as usize];
            let mut moved = false;
            for i in 0..3 {
                let a = self.pos(grid, tri.v[i]);
                let b = self.pos(grid, tri.v[(i + 1) % 3]);
                if orient2d(a, b, p) < 0 && tri.n[i] != NONE {
                    t = tri.n[i];
                    moved = true;
                    break;
                }
            }
            if !moved {
                return t;
            }
        }
        t
    }

    /// Bowyer-Watson insertion of the grid point `gi`, starting the cavity
    /// search from `seed` (which must contain the point). Returns the slots
    /// of the triangles that were created.
    fn insert(&mut self, grid: &Grid, gi: u32, seed: u32) -> Vec<u32> {
        let p = grid.coord(gi);
        let vid = self.verts.len() as u32;
        self.verts.push(gi);

        // 1. Collect the cavity: every triangle whose circumcircle contains
        // `p`. It is connected, so a flood fill through neighbours suffices.
        let mut cavity = vec![seed];
        let mut in_cavity = vec![false; self.tris.len()];
        in_cavity[seed as usize] = true;
        let mut cursor = 0;
        while cursor < cavity.len() {
            let t = cavity[cursor];
            cursor += 1;
            for i in 0..3 {
                let nb = self.tris[t as usize].n[i];
                if nb == NONE || in_cavity[nb as usize] {
                    continue;
                }
                let tri = &self.tris[nb as usize];
                let (a, b, c) = (
                    self.pos(grid, tri.v[0]),
                    self.pos(grid, tri.v[1]),
                    self.pos(grid, tri.v[2]),
                );
                if in_circle(a, b, c, p) > 0 {
                    in_cavity[nb as usize] = true;
                    cavity.push(nb);
                }
            }
        }

        // 2. The cavity boundary: edges not shared with another cavity
        // triangle. They come out oriented CCW around the cavity, so
        // `(a, b, p)` is CCW for an interior `p`.
        let mut boundary = Vec::new();
        for &t in &cavity {
            let tri = self.tris[t as usize].clone();
            for i in 0..3 {
                let outer = tri.n[i];
                if outer != NONE && in_cavity[outer as usize] {
                    continue;
                }
                boundary.push((tri.v[i], tri.v[(i + 1) % 3], outer));
            }
        }

        // 3. Retire the cavity and re-fan it from the new vertex. An edge
        // collinear with `p` would make a zero-area triangle — that is the
        // hull edge `p` landed on, and skipping it simply splits the hull
        // in two, which is exactly right.
        for &t in &cavity {
            self.tris[t as usize].alive = false;
            self.free.push(t);
        }

        let mut created = Vec::with_capacity(boundary.len());
        let mut fan = Vec::with_capacity(boundary.len());
        for &(a, b, outer) in &boundary {
            if orient2d(self.pos(grid, a), self.pos(grid, b), p) == 0 {
                continue;
            }
            let tri = Tri {
                v: [a, b, vid],
                n: [outer, NONE, NONE],
                cand: NONE,
                err: 0.0,
                alive: true,
            };
            let slot = match self.free.pop() {
                Some(slot) => {
                    self.tris[slot as usize] = tri;
                    slot
                }
                None => {
                    self.tris.push(tri);
                    self.tris.len() as u32 - 1
                }
            };
            created.push(slot);
            fan.push((a, b, slot));
        }

        // 4. Relink adjacency. Edge 0 of a new triangle faces the outer
        // neighbour, edge 1 `(b, vid)` faces the fan triangle starting at
        // `b`, and edge 2 `(vid, a)` the one ending at `a`. Anything left
        // unmatched is a hull edge.
        for &(a, b, slot) in &fan {
            let n1 = fan
                .iter()
                .find(|&&(oa, _, os)| oa == b && os != slot)
                .map_or(NONE, |&(_, _, os)| os);
            let n2 = fan
                .iter()
                .find(|&&(_, ob, os)| ob == a && os != slot)
                .map_or(NONE, |&(_, _, os)| os);
            self.tris[slot as usize].n[1] = n1;
            self.tris[slot as usize].n[2] = n2;

            // Point the outer neighbour back at us.
            let outer = self.tris[slot as usize].n[0];
            if outer != NONE {
                let otri = &mut self.tris[outer as usize];
                for i in 0..3 {
                    if otri.v[i] == b && otri.v[(i + 1) % 3] == a {
                        otri.n[i] = slot;
                        break;
                    }
                }
            }
        }

        created
    }

    /// Find the worst-approximated grid sample inside triangle `t`.
    ///
    /// Straightforward bounding-box rasterisation with the edge functions
    /// doubling as barycentric weights. Cost is proportional to the
    /// triangle's area, which shrinks quickly as insertion proceeds.
    fn compute_candidate(&mut self, grid: &Grid, t: u32) {
        let tri = &self.tris[t as usize];
        let (va, vb, vc) = (tri.v[0], tri.v[1], tri.v[2]);
        let (a, b, c) = (self.pos(grid, va), self.pos(grid, vb), self.pos(grid, vc));
        let area2 = orient2d(a, b, c);
        if area2 <= 0 {
            let tri = &mut self.tris[t as usize];
            tri.cand = NONE;
            tri.err = 0.0;
            return;
        }
        let (ha, hb, hc) = (
            self.height(grid, va),
            self.height(grid, vb),
            self.height(grid, vc),
        );
        let inv = 1.0 / area2 as f64;

        let min_x = a[0].min(b[0]).min(c[0]).max(0);
        let max_x = a[0].max(b[0]).max(c[0]).min(grid.nx as i32 - 1);
        let min_y = a[1].min(b[1]).min(c[1]).max(0);
        let max_y = a[1].max(b[1]).max(c[1]).min(grid.ny as i32 - 1);

        let mut best = NONE;
        let mut best_err = 0.0f32;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let q = [x, y];
                let w0 = orient2d(b, c, q);
                if w0 < 0 {
                    continue;
                }
                let w1 = orient2d(c, a, q);
                if w1 < 0 {
                    continue;
                }
                let w2 = orient2d(a, b, q);
                if w2 < 0 {
                    continue;
                }
                let lerp = (w0 as f64 * ha as f64 + w1 as f64 * hb as f64 + w2 as f64 * hc as f64)
                    * inv;
                let gi = grid.index(x as u32, y as u32);
                let err = (grid.height(gi) - lerp as f32).abs();
                if err > best_err {
                    best_err = err;
                    best = gi;
                }
            }
        }

        let tri = &mut self.tris[t as usize];
        tri.cand = best;
        tri.err = best_err;
    }
}

/// Douglas-Peucker over a line of samples. Splitting on the largest error
/// and tie-breaking on the lowest index makes this a pure function of the
/// height data, so the two chunks sharing a border independently derive the
/// *same* vertex set — which is what keeps the seam crack-free.
fn simplify_line(samples: &[f32], max_error: f32, out: &mut Vec<u32>) {
    fn recurse(samples: &[f32], lo: u32, hi: u32, max_error: f32, out: &mut Vec<u32>) {
        if hi <= lo + 1 {
            return;
        }
        let (a, b) = (samples[lo as usize], samples[hi as usize]);
        let span = (hi - lo) as f32;
        let mut best = 0u32;
        let mut best_err = 0.0f32;
        for i in lo + 1..hi {
            let t = (i - lo) as f32 / span;
            let err = (samples[i as usize] - (a + (b - a) * t)).abs();
            if err > best_err {
                best_err = err;
                best = i;
            }
        }
        if best_err > max_error {
            out.push(best);
            recurse(samples, lo, best, max_error, out);
            recurse(samples, best, hi, max_error, out);
        }
    }
    if samples.len() >= 2 {
        recurse(samples, 0, samples.len() as u32 - 1, max_error, out);
    }
}

/// Evenly-spread lattice positions along one axis, endpoints included, at
/// most `step` texels apart. A pure function of `(extent, step)` — two
/// chunks sharing a border have the same extent along it and therefore
/// derive the same border lattice, which is what keeps the seam crack-free.
fn lattice_positions(extent: u32, step: Option<u32>) -> Vec<u32> {
    match step {
        Some(s) if s < extent => {
            let count = extent.div_ceil(s);
            (0..=count).map(|i| i * extent / count).collect()
        }
        _ => vec![0, extent],
    }
}

/// Bring a chunk's triangulation within tolerance of its grid.
///
/// `border_error` is deliberately separate from `max_error`: it is the
/// *finest* tolerance across all detail levels, not this level's own.
/// The seam between two chunks is crack-free because both derive the same
/// vertices from the same samples — but only if they use the same tolerance
/// to do it. Two neighbours at different detail levels do not, so fitting
/// every border at the finest tolerance makes the shared edge identical no
/// matter which levels meet there. It costs little: a chunk's border is
/// `4 · CHUNK_SIZE` samples against `CHUNK_SIZE²` in the interior.
///
/// The curvature `lattice` (a sorted list of grid indices; see
/// [`Mapping::curvature_steps`] and `lattice_for_lod` in [`build`]) is
/// inserted first: it exists to bound world-space chord error, which no
/// height-space tolerance can see.
fn refine(
    chunk: &mut Chunk,
    grid: &Grid,
    lattice: &[u32],
    max_error: f32,
    border_error: f32,
) {
    use std::collections::{BinaryHeap, HashSet};

    let mut existing: HashSet<u32> = chunk.verts.iter().copied().collect();
    for &gi in lattice {
        if !existing.insert(gi) {
            continue;
        }
        let seed = chunk.locate(grid, grid.coord(gi), 0);
        chunk.insert(grid, gi, seed);
    }

    // Border vertices first: both chunks sharing a border derive the same
    // set from the same samples, so the seam matches exactly.
    let (mx, my) = (grid.nx - 1, grid.ny - 1);
    let mut border = Vec::new();
    let mut line = Vec::with_capacity(grid.nx.max(grid.ny) as usize);
    for (fixed, horizontal) in [(0, true), (my, true), (0, false), (mx, false)] {
        line.clear();
        let count = if horizontal { grid.nx } else { grid.ny };
        for i in 0..count {
            let gi = if horizontal {
                grid.index(i, fixed)
            } else {
                grid.index(fixed, i)
            };
            line.push(grid.height(gi));
        }
        let mut picks = Vec::new();
        simplify_line(&line, border_error, &mut picks);
        for i in picks {
            border.push(if horizontal {
                grid.index(i, fixed)
            } else {
                grid.index(fixed, i)
            });
        }
    }
    // Deterministic order keeps the whole build reproducible.
    border.sort_unstable();
    border.dedup();
    for gi in border {
        if existing.contains(&gi) {
            continue;
        }
        let seed = chunk.locate(grid, grid.coord(gi), 0);
        chunk.insert(grid, gi, seed);
    }

    // Every triangle remembers its own worst sample, so popping the global
    // worst needs no point location — we already know which triangle
    // contains it.
    for t in 0..chunk.tris.len() as u32 {
        if chunk.tris[t as usize].alive {
            chunk.compute_candidate(grid, t);
        }
    }
    let mut heap = BinaryHeap::new();
    for (t, tri) in chunk.tris.iter().enumerate() {
        if tri.alive && tri.cand != NONE {
            heap.push((tri.err.to_bits(), t as u32));
        }
    }

    // The TIN can never usefully exceed the source grid, so this is only a
    // runaway guard, not a quality knob.
    let max_vertices = grid.nx * grid.ny;
    while (chunk.verts.len() as u32) < max_vertices {
        let (bits, t) = match heap.pop() {
            Some(entry) => entry,
            None => break,
        };
        {
            // Stale entry: the slot was rewritten since it was queued.
            let tri = &chunk.tris[t as usize];
            if !tri.alive || tri.cand == NONE || tri.err.to_bits() != bits {
                continue;
            }
            if tri.err <= max_error {
                break;
            }
        }
        let cand = chunk.tris[t as usize].cand;
        for slot in chunk.insert(grid, cand, t) {
            chunk.compute_candidate(grid, slot);
            let tri = &chunk.tris[slot as usize];
            if tri.cand != NONE {
                heap.push((tri.err.to_bits(), slot));
            }
        }
    }
}

/// One LOD's worth of world-space geometry for one chunk.
#[derive(Default)]
struct LodMesh {
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

/// Turn a finished triangulation into world-space geometry.
fn emit_chunk(chunk: &Chunk, grid: &Grid, mapping: &Mapping) -> LodMesh {
    let mut out = LodMesh::default();
    let mut cache = vec![NONE; chunk.verts.len()];
    let flip = mapping.flip_winding();
    for tri in chunk.tris.iter().filter(|t| t.alive) {
        let mut ids = [0u32; 3];
        for (slot, &v) in ids.iter_mut().zip(tri.v.iter()) {
            let cached = &mut cache[v as usize];
            if *cached == NONE {
                let local = chunk.pos(grid, v);
                // Samples describe texel *cells*, so the vertex belongs at
                // the cell centre — matching how the shaders index the
                // terrain texture.
                let pos = mapping.embed(
                    (grid.x0 + local[0]) as f32 + 0.5,
                    (grid.y0 + local[1]) as f32 + 0.5,
                    chunk.height(grid, v),
                );
                *cached = out.vertices.len() as u32;
                out.vertices.push(pos);
            }
            *slot = *cached;
        }
        if flip {
            ids.swap(1, 2);
        }
        out.indices.extend_from_slice(&ids);
    }
    out
}

/// One chunk's geometry: all LODs packed into a single vertex + index list.
///
/// Per-chunk buffers rather than one buffer for the level: a level-sized
/// buffer runs past `max_buffer_size` on WebGL-class limits long before the
/// geometry itself is unreasonable, and per-chunk granularity is what makes
/// frustum culling and LOD selection possible at draw time.
pub struct ChunkBuffers {
    pub vertices: Vec<[f32; 3]>,
    /// Indices into `vertices`; each LOD's triangles are contiguous.
    pub indices: Vec<u32>,
    /// `(first index, index count)` per LOD, finest first.
    pub lods: Vec<(u32, u32)>,
    /// Number of leading entries of `vertices` used by LOD 0 — the slice the
    /// physics trimesh needs.
    pub lod0_vertex_count: u32,
    /// World-space bounds, for frustum culling and LOD distance.
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl ChunkBuffers {
    fn new(per_lod: Vec<LodMesh>) -> Self {
        let mut buffers = ChunkBuffers {
            vertices: Vec::new(),
            indices: Vec::new(),
            lods: Vec::with_capacity(per_lod.len()),
            lod0_vertex_count: per_lod.first().map_or(0, |m| m.vertices.len() as u32),
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        };
        for mesh in per_lod {
            let base = buffers.vertices.len() as u32;
            let first = buffers.indices.len() as u32;
            buffers
                .indices
                .extend(mesh.indices.iter().map(|i| i + base));
            for v in &mesh.vertices {
                for ((lo, hi), &c) in buffers.min.iter_mut().zip(buffers.max.iter_mut()).zip(v) {
                    *lo = lo.min(c);
                    *hi = hi.max(c);
                }
            }
            buffers.vertices.extend(mesh.vertices);
            buffers
                .lods
                .push((first, buffers.indices.len() as u32 - first));
        }
        buffers
    }

    pub fn center(&self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// The finest-LOD triangle list, for building a physics trimesh.
    pub fn lod0(&self) -> (&[[f32; 3]], &[u32]) {
        let (first, count) = self.lods[0];
        (
            &self.vertices[..self.lod0_vertex_count as usize],
            &self.indices[first as usize..(first + count) as usize],
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub vertices: usize,
    pub triangles: usize,
    /// Total triangles per LOD across all chunks, finest first.
    /// `lod_triangles[0] == triangles`.
    pub lod_triangles: [usize; LOD_COUNT],
    pub source_texels: usize,
    pub max_error: f32,
}

/// The whole terrain as chunked render/physics geometry.
pub struct TerrainMesh {
    pub mapping: Mapping,
    pub chunks: Vec<ChunkBuffers>,
    pub stats: Stats,
}

/// Build the TIN for a height map.
///
/// `quality` in `0..=1` — see [`max_error_for_quality`].
pub fn build(
    alpha: &[u8],
    width: u32,
    height: u32,
    config: &MapConfig,
    quality: f32,
) -> TerrainMesh {
    profiling::scope!("Tin::build");
    assert_eq!(alpha.len(), (width as usize) * (height as usize));
    let mapping = Mapping::new(config, width, height);
    let max_error = max_error_for_quality(quality);

    // Chunks own the texels `[x0 ..= x0 + w]`, sharing their border
    // row/column with the neighbour. The wrapping direction covers
    // `[0 ..= width]` (the far border re-samples column 0, closing the
    // seam); the clamping direction stops at the last texel.
    let spans_of = |total: i32| {
        let step = CHUNK_SIZE as i32;
        let mut spans = Vec::new();
        let mut at = 0;
        while at < total {
            spans.push((at, (total - at).min(step) as u32));
            at += step;
        }
        spans
    };
    let x_spans = spans_of(width as i32);
    let y_spans = if mapping.wrap_y() {
        spans_of(height as i32)
    } else {
        spans_of(height as i32 - 1)
    };
    let mut origins = Vec::with_capacity(x_spans.len() * y_spans.len());
    for &(y, h) in &y_spans {
        for &(x, w) in &x_spans {
            origins.push((x, y, w, h));
        }
    }

    let tol_world = mapping.tol_world(max_error);

    let build_one = |&(x, y, w, h): &(i32, i32, u32, u32)| -> ChunkBuffers {
        let grid = Grid::new(&mapping, alpha, x, y, w, h);

        // Curvature lattice, as sorted grid indices, for one LOD.
        //
        // The border lines are always populated at the *finest* spacing —
        // the same rule the height fit uses for `border_error`, and for the
        // same reason: two neighbours drawn at different LODs must derive
        // the identical vertex set on their shared line or the seam cracks.
        // (The interior spacings of different LODs are not nested subsets of
        // each other, so pinning the borders is what makes mixing safe.)
        // Only the interior coarsens with the LOD's tolerance: doubling the
        // tolerance widens the spacing by √2.
        let (finest_x, finest_y) = {
            let (us, vs) = mapping.curvature_steps(tol_world, y, h);
            (lattice_positions(w, us), lattice_positions(h, vs))
        };
        let lattice_for_lod = |k: usize| -> Vec<u32> {
            let tol_k = mapping.tol_world(max_error * (1 << k) as f32);
            let (us, vs) = mapping.curvature_steps(tol_k, y, h);
            let inner_x = lattice_positions(w, us);
            let inner_y = lattice_positions(h, vs);
            let mut points = Vec::new();
            for &lx in &finest_x {
                points.push(grid.index(lx, 0));
                points.push(grid.index(lx, h));
            }
            for &ly in &finest_y {
                points.push(grid.index(0, ly));
                points.push(grid.index(w, ly));
            }
            for &ly in &inner_y {
                if ly == 0 || ly == h {
                    continue;
                }
                for &lx in &inner_x {
                    if lx == 0 || lx == w {
                        continue;
                    }
                    points.push(grid.index(lx, ly));
                }
            }
            // Deterministic order keeps the whole build reproducible.
            points.sort_unstable();
            points.dedup();
            points
        };

        // Each LOD is an independent fit at a doubled tolerance. They could
        // share work — the coarse vertex sets are prefixes of the fine one —
        // but refitting from scratch is cheap (the coarse levels converge in
        // a fraction of the insertions) and keeps every level a genuine
        // Delaunay triangulation.
        let mut per_lod = Vec::with_capacity(LOD_COUNT);
        for k in 0..LOD_COUNT {
            let mut chunk = Chunk::new(&grid);
            refine(
                &mut chunk,
                &grid,
                &lattice_for_lod(k),
                max_error * (1 << k) as f32,
                max_error,
            );
            per_lod.push(emit_chunk(&chunk, &grid, &mapping));
        }
        // The mesh bulges between vertices by up to the curvature-lattice
        // tolerance plus the coarsest LOD's height slack; pad the culling
        // AABB by a conservative multiple of both.
        let mut buffers = ChunkBuffers::new(per_lod);
        let pad = 2.0 * tol_world
            + mapping.tol_world(max_error * (1 << (LOD_COUNT - 1)) as f32)
            + 0.05;
        for k in 0..3 {
            buffers.min[k] -= pad;
            buffers.max[k] += pad;
        }
        buffers
    };

    #[cfg(not(target_arch = "wasm32"))]
    let chunks: Vec<ChunkBuffers> = {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(origins.len().max(1));
        let per = origins.len().div_ceil(workers);
        let mut results: Vec<Option<ChunkBuffers>> = Vec::new();
        results.resize_with(origins.len(), || None);
        std::thread::scope(|s| {
            for (chunk_in, chunk_out) in origins.chunks(per).zip(results.chunks_mut(per)) {
                s.spawn(move || {
                    for (origin, out) in chunk_in.iter().zip(chunk_out.iter_mut()) {
                        *out = Some(build_one(origin));
                    }
                });
            }
        });
        results.into_iter().map(Option::unwrap).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let chunks: Vec<ChunkBuffers> = origins.iter().map(build_one).collect();

    let mut stats = Stats {
        source_texels: (width as usize) * (height as usize),
        max_error,
        ..Default::default()
    };
    for chunk in &chunks {
        // Headline stats describe LOD 0 — the mesh as actually drawn up close.
        stats.vertices += chunk.lod0_vertex_count as usize;
        stats.triangles += chunk.lods[0].1 as usize / 3;
        for (total, &(_, count)) in stats.lod_triangles.iter_mut().zip(&chunk.lods) {
            *total += count as usize / 3;
        }
    }
    log::info!(
        "Terrain TIN at quality {}: {} chunks, {} vertices, {} triangles from {} texels \
         ({:.1}x fewer triangles, max error {:.2} height bytes), LOD triangles {:?}",
        quality,
        chunks.len(),
        stats.vertices,
        stats.triangles,
        stats.source_texels,
        2.0 * stats.source_texels as f32 / stats.triangles.max(1) as f32,
        stats.max_error,
        stats.lod_triangles,
    );

    TerrainMesh {
        mapping,
        chunks,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_config(shape: WorldShape) -> MapConfig {
        MapConfig {
            radius: 10.0..15.0,
            length: 200.0,
            density: 1.0,
            shape,
        }
    }

    /// Rolling hills over the whole map.
    fn hills(width: u32, height: u32) -> Vec<u8> {
        let mut alpha = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let a = 100.0
                    + 60.0 * (x as f32 * 0.2).sin()
                    + 50.0 * (y as f32 * 0.15).sin();
                alpha[(y * width + x) as usize] = a as u8;
            }
        }
        alpha
    }

    fn bumpy_grid(mapping: &Mapping, alpha: &[u8], size: u32) -> Grid {
        Grid::new(mapping, alpha, 0, 0, size, size)
    }

    /// Every alive triangle must be CCW, and adjacency must be symmetric.
    fn check_invariants(chunk: &Chunk, grid: &Grid) {
        for (t, tri) in chunk.tris.iter().enumerate() {
            if !tri.alive {
                continue;
            }
            let (a, b, c) = (
                chunk.pos(grid, tri.v[0]),
                chunk.pos(grid, tri.v[1]),
                chunk.pos(grid, tri.v[2]),
            );
            assert!(orient2d(a, b, c) > 0, "triangle {} is not CCW", t);

            for i in 0..3 {
                let nb = tri.n[i];
                if nb == NONE {
                    continue;
                }
                let other = &chunk.tris[nb as usize];
                assert!(other.alive, "triangle {} links to a dead neighbour", t);
                let (e0, e1) = (tri.v[i], tri.v[(i + 1) % 3]);
                let found = (0..3).any(|j| {
                    other.v[j] == e1 && other.v[(j + 1) % 3] == e0 && other.n[j] == t as u32
                });
                assert!(found, "adjacency of {} across edge {} is not mutual", t, i);
            }
        }
    }

    /// Delaunay's defining property: no vertex inside any circumcircle.
    fn check_delaunay(chunk: &Chunk, grid: &Grid) {
        for tri in chunk.tris.iter().filter(|t| t.alive) {
            let (a, b, c) = (
                chunk.pos(grid, tri.v[0]),
                chunk.pos(grid, tri.v[1]),
                chunk.pos(grid, tri.v[2]),
            );
            for v in 0..chunk.verts.len() as u32 {
                if tri.v.contains(&v) {
                    continue;
                }
                assert!(
                    in_circle(a, b, c, chunk.pos(grid, v)) <= 0,
                    "vertex {} violates the empty-circumcircle property",
                    v
                );
            }
        }
    }

    #[test]
    fn refine_keeps_triangulation_valid_and_delaunay() {
        let (w, h) = (48u32, 48u32);
        let alpha = hills(w + 1, h + 1);
        let mapping = Mapping::new(&map_config(WorldShape::Cylinder), w + 1, h + 1);
        let grid = bumpy_grid(&mapping, &alpha, w);
        let picks = lattice_positions(w, Some(16));
        let mut lattice = Vec::new();
        for &ly in &picks {
            for &lx in &picks {
                lattice.push(grid.index(lx, ly));
            }
        }
        lattice.sort_unstable();
        let mut chunk = Chunk::new(&grid);
        refine(&mut chunk, &grid, &lattice, 2.0, 2.0);
        check_invariants(&chunk, &grid);
        check_delaunay(&chunk, &grid);
        assert!(chunk.verts.len() > 4, "the hills must force insertions");
    }

    #[test]
    fn refine_meets_the_error_tolerance() {
        let (w, h) = (48u32, 48u32);
        let alpha = hills(w + 1, h + 1);
        let mapping = Mapping::new(&map_config(WorldShape::Cylinder), w + 1, h + 1);
        let grid = bumpy_grid(&mapping, &alpha, w);
        let mut chunk = Chunk::new(&grid);
        let tolerance = 3.0;
        refine(&mut chunk, &grid, &[], tolerance, tolerance);
        for t in 0..chunk.tris.len() as u32 {
            if chunk.tris[t as usize].alive {
                chunk.compute_candidate(&grid, t);
                assert!(
                    chunk.tris[t as usize].err <= tolerance,
                    "triangle {} still has error {}",
                    t,
                    chunk.tris[t as usize].err
                );
            }
        }
    }

    #[test]
    fn flat_map_stays_coarse() {
        let (w, h) = (256u32, 256u32);
        let alpha = vec![128u8; (w * h) as usize];
        let mesh = build(&alpha, w, h, &map_config(WorldShape::Cylinder), 1.0);
        // Flat data needs no insertions beyond the curvature lattice, which
        // is far sparser than the source grid.
        assert!(
            mesh.stats.triangles < mesh.stats.source_texels / 4,
            "flat map should mesh far below grid resolution: {} triangles from {} texels",
            mesh.stats.triangles,
            mesh.stats.source_texels
        );
    }

    #[test]
    fn build_is_deterministic() {
        let (w, h) = (192u32, 192u32);
        let alpha = hills(w, h);
        let config = map_config(WorldShape::Cylinder);
        let a = build(&alpha, w, h, &config, 0.75);
        let b = build(&alpha, w, h, &config, 0.75);
        assert_eq!(a.chunks.len(), b.chunks.len());
        for (ca, cb) in a.chunks.iter().zip(&b.chunks) {
            assert_eq!(ca.indices, cb.indices);
            assert_eq!(ca.vertices, cb.vertices);
            assert_eq!(ca.lods, cb.lods);
        }
    }

    #[test]
    fn theta_seam_is_crack_free() {
        // The wrap column: the last chunk's right border re-samples column 0,
        // so both sides of the θ = 0 seam must independently derive the same
        // vertex heights along it — and therefore the same world positions.
        let (w, h) = (256u32, 96u32);
        let alpha = hills(w, h);
        let config = map_config(WorldShape::Cylinder);
        let mesh = build(&alpha, w, h, &config, 1.0);

        // Collect world vertices lying on the seam (θ = 0.5 texel) from both
        // ends of the map and compare as sets.
        let seam_x = 0.5 / w as f32 * std::f32::consts::TAU; // θ of column 0's centre
        let on_seam = |v: &[f32; 3]| {
            let theta = v[1].atan2(v[0]);
            (theta - seam_x).abs() < 1e-4 || (theta - seam_x + std::f32::consts::TAU).abs() < 1e-4
        };
        let mut seam_verts: Vec<[i32; 2]> = mesh
            .chunks
            .iter()
            .flat_map(|c| c.vertices.iter())
            .filter(|v| on_seam(v))
            .map(|v| [(v[2] * 1000.0) as i32, (v[0].hypot(v[1]) * 1000.0) as i32])
            .collect();
        seam_verts.sort_unstable();
        assert!(!seam_verts.is_empty(), "no vertices found on the θ seam");
        // Every seam vertex appears exactly twice: once from the chunk on
        // each side. An odd count means one side placed a vertex the other
        // did not — a crack.
        seam_verts.chunk_by(|a, b| a == b).for_each(|group| {
            assert_eq!(
                group.len() % 2,
                0,
                "seam vertex {:?} is not mirrored on both sides",
                group[0]
            );
        });
    }

    #[test]
    fn cylinder_vertices_stay_in_the_radial_shell() {
        let (w, h) = (128u32, 128u32);
        let alpha = hills(w, h);
        let config = map_config(WorldShape::Cylinder);
        let mesh = build(&alpha, w, h, &config, 0.75);
        for chunk in &mesh.chunks {
            for v in &chunk.vertices {
                let r = v[0].hypot(v[1]);
                assert!(
                    (config.radius.start - 1e-3..config.radius.end + 1e-3).contains(&r),
                    "vertex radius {} outside the shell",
                    r
                );
                assert!(v[2].abs() <= 0.5 * config.length + 1e-3);
            }
        }
    }

    #[test]
    fn torus_vertices_ring_the_major_circle() {
        let (w, h) = (128u32, 256u32);
        let alpha = hills(w, h);
        let config = map_config(WorldShape::Torus);
        let mesh = build(&alpha, w, h, &config, 0.75);
        let big_r = mesh.mapping.major_radius();
        assert!(
            big_r > config.radius.end,
            "test map too short for a torus: R = {}",
            big_r
        );
        for chunk in &mesh.chunks {
            for v in &chunk.vertices {
                let ring_dist = v[0].hypot(v[1]) - big_r;
                let tube_r = ring_dist.hypot(v[2]);
                assert!(
                    (config.radius.start - 1e-2..config.radius.end + 1e-2).contains(&tube_r),
                    "vertex tube radius {} outside the shell",
                    tube_r
                );
            }
        }
    }

    #[test]
    fn triangles_face_outward_on_every_shape() {
        for shape in [WorldShape::Cylinder, WorldShape::Sphere, WorldShape::Torus] {
            let (w, h) = (64u32, 128u32);
            let alpha = vec![128u8; (w * h) as usize];
            let config = map_config(shape);
            let mesh = build(&alpha, w, h, &config, 1.0);
            let big_r = mesh.mapping.major_radius();
            let mut checked = 0;
            for chunk in &mesh.chunks {
                let (verts, indices) = chunk.lod0();
                for tri in indices.chunks(3) {
                    let p = |i: usize| {
                        let v = verts[tri[i] as usize];
                        nalgebra::Vector3::new(v[0], v[1], v[2])
                    };
                    let (a, b, c) = (p(0), p(1), p(2));
                    let normal = (b - a).cross(&(c - a));
                    if normal.norm() < 1e-6 {
                        continue; // degenerate (sphere pole)
                    }
                    let centroid = (a + b + c) / 3.0;
                    let outward = match shape {
                        WorldShape::Cylinder => {
                            nalgebra::Vector3::new(centroid.x, centroid.y, 0.0)
                        }
                        WorldShape::Sphere => centroid,
                        WorldShape::Torus => {
                            let rxy = centroid.xy().norm().max(1e-6);
                            let ring = nalgebra::Vector3::new(
                                centroid.x * big_r / rxy,
                                centroid.y * big_r / rxy,
                                0.0,
                            );
                            centroid - ring
                        }
                    };
                    assert!(
                        normal.dot(&outward) > 0.0,
                        "{:?}: inward-facing triangle at {:?}",
                        shape,
                        centroid
                    );
                    checked += 1;
                }
            }
            assert!(checked > 0);
        }
    }

    #[test]
    fn lods_shrink_and_share_the_border_fit() {
        let (w, h) = (256u32, 256u32);
        let alpha = hills(w, h);
        let config = map_config(WorldShape::Cylinder);
        let mesh = build(&alpha, w, h, &config, 1.0);
        for chunk in &mesh.chunks {
            assert_eq!(chunk.lods.len(), LOD_COUNT);
            for pair in chunk.lods.windows(2) {
                assert!(
                    pair[1].1 <= pair[0].1,
                    "coarser LOD has more indices: {:?}",
                    chunk.lods
                );
            }
        }
    }

    #[test]
    fn quality_knob_trades_triangles_for_error() {
        let (w, h) = (192u32, 192u32);
        let alpha = hills(w, h);
        let config = map_config(WorldShape::Cylinder);
        let fine = build(&alpha, w, h, &config, 1.0);
        let coarse = build(&alpha, w, h, &config, 0.25);
        assert!(fine.stats.triangles > coarse.stats.triangles);
    }

    /// Radial distance of a world point in the shape's own parameterisation
    /// — the coordinate the ground radius is measured along.
    fn radial_distance(shape: WorldShape, major_radius: f32, p: [f32; 3]) -> f32 {
        match shape {
            WorldShape::Cylinder => p[0].hypot(p[1]),
            WorldShape::Sphere => (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt(),
            WorldShape::Torus => (p[0].hypot(p[1]) - major_radius).hypot(p[2]),
        }
    }

    /// The curvature lattice's contract: on a *flat* height map (where the
    /// greedy fit contributes nothing) every triangle of every LOD must stay
    /// within that LOD's world tolerance of the true curved surface.
    ///
    /// The deviation is two-sided: convex-outward directions sag chords
    /// *inward*, but the torus's inner ring is concave along the major
    /// direction, so chords bulge *outward* there — by less (`ρ = R - r`
    /// governs it) than the outer ring's inward sag (`ρ = R + r`, which is
    /// what sizes the budget), so one magnitude limit covers both.
    #[test]
    fn chord_error_stays_within_each_lods_tolerance() {
        for shape in [WorldShape::Cylinder, WorldShape::Sphere, WorldShape::Torus] {
            let (w, h) = (128u32, 256u32);
            let alpha = vec![128u8; (w * h) as usize];
            let mut config = map_config(shape);
            // Long enough that the torus does not self-intersect.
            config.length = 600.0;
            let quality = 1.0;
            let mesh = build(&alpha, w, h, &config, quality);
            let big_r = mesh.mapping.major_radius();
            let ground_r = mesh.mapping.ground_radius(128.0 / 255.0);

            for k in 0..LOD_COUNT {
                let tol_k = mesh
                    .mapping
                    .tol_world(max_error_for_quality(quality) * (1 << k) as f32);
                // Small-angle bounds and f32 embedding both eat a little
                // margin; 1.5x is comfortably above what they cost while
                // still far below the 4x that a missing budget split (or a
                // missing pole densification) produces.
                let limit = 1.5 * tol_k + 1e-3;
                let mut worst = 0.0f32;
                for chunk in &mesh.chunks {
                    let (first, count) = chunk.lods[k];
                    let indices = &chunk.indices[first as usize..(first + count) as usize];
                    for tri in indices.chunks(3) {
                        let v = |i: usize| chunk.vertices[tri[i] as usize];
                        let (a, b, c) = (v(0), v(1), v(2));
                        let mid = |p: [f32; 3], q: [f32; 3]| {
                            [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])]
                        };
                        let centroid = [
                            (a[0] + b[0] + c[0]) / 3.0,
                            (a[1] + b[1] + c[1]) / 3.0,
                            (a[2] + b[2] + c[2]) / 3.0,
                        ];
                        for p in [mid(a, b), mid(b, c), mid(c, a), centroid] {
                            let dev = ground_r - radial_distance(shape, big_r, p);
                            worst = worst.max(dev.abs());
                        }
                    }
                }
                assert!(
                    worst <= limit,
                    "{:?} LOD{}: worst chord deviation {} exceeds tolerance {} (limit {})",
                    shape,
                    k,
                    worst,
                    tol_k,
                    limit
                );
            }
        }
    }

    /// The interior lattice must coarsen with the LOD tolerance — that is
    /// what lets far-away flat-but-curved terrain actually get cheaper. The
    /// borders stay pinned at the finest spacing, so the counts shrink
    /// rather than collapse.
    #[test]
    fn coarser_lods_thin_the_curvature_lattice() {
        for shape in [WorldShape::Sphere, WorldShape::Torus] {
            let (w, h) = (128u32, 256u32);
            let alpha = vec![128u8; (w * h) as usize];
            let mut config = map_config(shape);
            config.length = 600.0;
            let mesh = build(&alpha, w, h, &config, 1.0);
            let lods = mesh.stats.lod_triangles;
            assert!(
                lods[LOD_COUNT - 1] < lods[0],
                "{:?}: coarsest LOD should carry fewer lattice triangles: {:?}",
                shape,
                lods
            );
        }
    }
}
