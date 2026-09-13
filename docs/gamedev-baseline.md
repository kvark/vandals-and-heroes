# GameDev baseline

Shared notes for kvark game projects: stacks, assets, headless smoke, and CI expectations. Owned in this repo so the Blade titles stay in one place.

Living notes (edit day-to-day): https://hackmd.io/s/HkMqq2btze

## Rooms and ownership

**GameDev room (Blade stack only)** — Overseer coordinates; Vanger / `vange-rs` is **not** in this room (talk 1:1).

| Agent | Owns |
|-------|------|
| Eva | [`geofront`](https://github.com/kvark/geofront) |
| Goat | [`claymore-blade`](https://github.com/kvark/claymore-blade) |
| Hero | [`vandals-and-heroes`](https://github.com/kvark/vandals-and-heroes) |
| Sonic | [`redline`](https://github.com/kvark/redline) |
| Twist | [`screech`](https://github.com/kvark/screech) — Rusted-Metal / Twisted Metal 2–style on Blade |
| Overseer | CI greenkeeping, review/merge, playtests, north-star nudges |

**Separate track:** Vanger owns [`vange-rs`](https://github.com/kvark/vange-rs) (wgpu + persistent multiplayer). Coordinate outside GameDev.

## North stars

| Project | North star |
|---------|------------|
| geofront | Evangelion look and mechanics (`ideas/game/eva.md`) |
| claymore-blade | Claymore anime |
| vandals-and-heroes | Original vehicular combat RPG in the Vangers universe (sibling to vange-rs, not a clone) |
| redline | Original futuristic planet-circuit racing (F-Zero/Wipeout energy; Mars flagship; physics-first ribbon) |
| screech | Rusted-Metal / Twisted Metal 2–style vehicular combat on Blade |
| vange-rs *(outside GameDev room)* | Vangers (1998) remake fidelity (OSS https://github.com/KranX/Vangers); **forward focus: persistent multiplayer** |

## Projects (Blade family — this room)

| Repo | Graphics | Physics / other | Assets | Headless / local | Web |
|------|----------|-----------------|--------|------------------|-----|
| [`vandals-and-heroes`](https://github.com/kvark/vandals-and-heroes) | Blade | Rapier, Winit, Choir | Git LFS | lavapipe | WebGL2 + Pages |
| [`redline`](https://github.com/kvark/redline) | Blade | Rapier joints | Git LFS (Kenney CC0) | lavapipe / Xvfb smoke + scripted drives | WASM + Pages |
| [`geofront`](https://github.com/kvark/geofront) | Blade (pin by **git rev**; see blade#389) | — | Quaternius / LFS-style GLBs; `scripts/fetch-shaders.sh` | `GEOFRONT_AUTOPLAY=1` under Xvfb + lavapipe | WASM |
| [`claymore-blade`](https://github.com/kvark/claymore-blade) | Blade | — (hex hunt, no Rapier) | Kenney CC0 + 2D art (no LFS GLBs) | lavapipe | Pages; see CI note below |
| [`screech`](https://github.com/kvark/screech) | Blade | (TBD — first vehicle slice) | TBD | Rasterizer default; xvfb+lavapipe (`SCREECH_RT=1` for RT) | TBD |

Engine: [`blade`](https://github.com/kvark/blade) — shared graphics stack; pin by git rev from game Cargo.toml.

## Local defaults (Blade)

- Prefer **lavapipe** (Mesa software Vulkan) for headless / no-GPU iteration.
- Use **Xvfb** when a display is required (`xvfb-run -a …`).
- `git lfs pull` (or project equivalent) before native or WASM builds that embed models.
- Release builds for playable framerate; debug when you need assertions.

## Web / WASM

- Target `wasm32-unknown-unknown`, `wasm-bindgen-cli` version matching `Cargo.lock`.
- WebGL2 browsers; Blade needs GLES-friendly shader linking on wasm (see per-repo `.cargo` / Blade notes).
- Pages deploys from `main` are common; they are **not** a substitute for PR checks.

## CI expectations (Blade)

| Check | Expectation |
|-------|-------------|
| Native tests / clippy / build | yes |
| Headless smoke (Xvfb ± lavapipe / autoplay) | preferred |
| **WASM on PRs** | **required** — do not rely only on the Pages job (Claymore today: native + GLES on PR, wasm only on Pages; bring wasm onto PRs) |
| Pages deploy | on `main` where enabled |

Merge policy (Overseer): CI green + reviewed + no big pivots + LOC proportional → rebase-merge; squash only a trailing CI-fix commit. Rebases force-push the **same** PR branch.

## Open follow-ups

- Land a wasm PR check on Blade repos that still gate web only via Pages (Claymore first).
- Keep Blade pins explicit (git rev) when tracking fixes such as blade#389.
- vange-rs notes live with Vanger (not this room’s day-to-day board).
