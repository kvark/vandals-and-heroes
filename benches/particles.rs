//! Particle-system benchmarks. Quantifies the three dominant costs of the
//! debug snow at 2000 particles:
//!
//! - `physics_step` — `Physics::update_gravity` + `Physics::step` on a flat
//!   cylindrical heightfield with N ball colliders falling. This is the
//!   *rapier* cost. Currently dominates.
//! - `snow_serial` — fetch every body's pose serially. Proxy for what
//!   `Snow::update` Phase 1 does.
//! - `snow_parallel` — pose fetch + per-particle bookkeeping fanned out to
//!   Choir workers with the same `init_multi` + disjoint-slice pattern that
//!   `Snow::update` uses. Measures whether the parallelisation actually
//!   helps at this N.
//!
//! Run with `cargo bench --bench particles -- --quick` for a fast pass.
//! Results land under `target/criterion/`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rapier3d::{
    dynamics::{RigidBodyBuilder, RigidBodyHandle},
    geometry::ColliderBuilder,
    math::Pose,
};
use std::sync::Arc;
use std::time::Duration;
use vandals_and_heroes::{Physics, PhysicsBodyHandle, TerrainBody, config};

const TERRAIN_WIDTH: u32 = 64;
const TERRAIN_HEIGHT: u32 = 256;
const TERRAIN_RADIUS_START: f32 = 10.0;
const TERRAIN_RADIUS_END: f32 = 20.0;
const TERRAIN_LENGTH: f32 = 100.0;
/// Counts to sweep. Cover the actual production case (2000) plus a few
/// smaller points so we can see the scaling curve.
const COUNTS: &[usize] = &[200, 500, 1000, 2000];

fn build_scene(particle_count: usize) -> (Physics, TerrainBody, Vec<RigidBodyHandle>) {
    let mut physics = Physics::default();
    let alpha = vec![128u8; (TERRAIN_WIDTH * TERRAIN_HEIGHT) as usize];
    let cfg = config::Map {
        radius: TERRAIN_RADIUS_START..TERRAIN_RADIUS_END,
        length: TERRAIN_LENGTH,
        density: 10.0,
        is_sphere: false,
    };
    let terrain = physics.create_terrain(&cfg, alpha, TERRAIN_WIDTH, TERRAIN_HEIGHT);

    let mut bodies = Vec::with_capacity(particle_count);
    let r_spawn = TERRAIN_RADIUS_END + 0.05;
    let mut rng_state: u64 = 0xC0FF_EE00_BAAD_F00D;
    for _ in 0..particle_count {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let theta =
            (rng_state >> 32) as u32 as f32 / (u32::MAX as f32 + 1.0) * std::f32::consts::TAU;
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let z = ((rng_state >> 32) as u32 as f32 / (u32::MAX as f32 + 1.0) - 0.5)
            * (TERRAIN_LENGTH * 0.5);
        let pose = Pose::from_translation(rapier3d::math::Vec3::new(
            r_spawn * theta.cos(),
            r_spawn * theta.sin(),
            z,
        ));
        let body = RigidBodyBuilder::dynamic()
            .pose(pose)
            .linear_damping(0.6)
            .angular_damping(0.2)
            .build();
        let collider = ColliderBuilder::ball(0.05)
            .density(0.2)
            .friction(0.5)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle, ..
        } = physics.add_rigid_body(body, vec![collider]);
        bodies.push(rigid_body_handle);
    }

    // Settle for a fraction of a second so the first step in the bench isn't
    // dominated by the initial wave of contacts being created.
    for _ in 0..30 {
        physics.update_gravity(&terrain);
        physics.step();
    }

    (physics, terrain, bodies)
}

fn bench_physics_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_step");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for &n in COUNTS {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let (mut physics, terrain, _bodies) = build_scene(n);
            b.iter(|| {
                physics.update_gravity(&terrain);
                physics.step();
            });
        });
    }
    group.finish();
}

fn bench_snow_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("snow_serial");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for &n in COUNTS {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let (physics, _terrain, bodies) = build_scene(n);
            let mut transforms: Vec<nalgebra::Isometry3<f32>> =
                vec![nalgebra::Isometry3::identity(); n];
            let mut age_ticks = vec![0u32; n];
            let lifetime_ticks: Vec<u32> = (0..n as u32).map(|i| 1200 + i).collect();
            b.iter(|| {
                for (i, &h) in bodies.iter().enumerate() {
                    transforms[i] = physics.get_transform(h);
                    age_ticks[i] = age_ticks[i].saturating_add(1);
                    if age_ticks[i] >= lifetime_ticks[i] {
                        age_ticks[i] = 0;
                    }
                }
            });
        });
    }
    group.finish();
}

/// `*mut T` wrapper that lets us share disjoint slice writes across Choir
/// workers. Mirrors the one in `bin/game/snow.rs`. The helper methods
/// (rather than direct `.0` access) force whole-struct capture into the
/// `move` closure under Rust 2021/2024 fine-grained closure captures.
#[derive(Copy, Clone)]
struct SyncSlice<T>(*mut T);
unsafe impl<T> Send for SyncSlice<T> {}
unsafe impl<T> Sync for SyncSlice<T> {}

impl<T> SyncSlice<T> {
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self, i: usize) -> &'static mut T {
        unsafe { &mut *self.0.add(i) }
    }
    #[inline]
    unsafe fn read(&self, i: usize) -> T
    where
        T: Copy,
    {
        unsafe { *self.0.add(i) }
    }
}

fn bench_snow_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("snow_parallel");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for &n in COUNTS {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let (physics, _terrain, bodies) = build_scene(n);
            let choir = choir::Choir::new();
            let _workers: Vec<_> = (0..4).map(|i| choir.add_worker(&format!("w{i}"))).collect();
            let mut transforms: Vec<nalgebra::Isometry3<f32>> =
                vec![nalgebra::Isometry3::identity(); n];
            let mut age_ticks = vec![0u32; n];
            let lifetime_ticks: Vec<u32> = (0..n as u32).map(|i| 1200 + i).collect();
            b.iter(|| {
                // Phase 1: serial pose snapshot.
                for (i, &h) in bodies.iter().enumerate() {
                    transforms[i] = physics.get_transform(h);
                }
                // Phase 2: parallel age + lifetime check on disjoint chunks.
                let workers: u32 = 4;
                let chunk = n.div_ceil(workers as usize);
                let ages_p = SyncSlice(age_ticks.as_mut_ptr());
                let lifetimes_p = SyncSlice(lifetime_ticks.as_ptr() as *mut u32);
                let task = choir
                    .spawn("snow_parallel_bench")
                    .init_multi(workers, move |_, worker_idx| {
                        let start = (worker_idx as usize) * chunk;
                        let end = ((worker_idx as usize + 1) * chunk).min(n);
                        for i in start..end {
                            unsafe {
                                let age = ages_p.get_mut(i);
                                *age = age.saturating_add(1);
                                let lifetime = lifetimes_p.read(i);
                                if *age >= lifetime {
                                    *age = 0;
                                }
                            }
                        }
                    })
                    .run();
                let _ = task.join();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_physics_step,
    bench_snow_serial,
    bench_snow_parallel
);
criterion_main!(benches);

// Suppress an unused-Arc warning on systems where `Arc::new` short-circuits.
const _: fn() = || {
    let _: Arc<u8> = Arc::new(0);
};
