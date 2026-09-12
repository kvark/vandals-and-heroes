# GameDev baseline

Shared notes for kvark game projects: stacks, assets, headless smoke, and CI expectations. Owned in this repo so the Blade titles and the wgpu outlier stay in one place.


## North stars

| Project | North star |
|---------|------------|
| geofront | Evangelion look and mechanics (`ideas/game/eva.md`) |
| claymore-blade | Claymore anime |
| vange-rs | Vangers (1998) remake; OSS https://github.com/KranX/Vangers |
| vandals-and-heroes | Original vehicular combat RPG in the Vangers universe (sibling to vange-rs, not a clone) |
| redline | Original futuristic planet-circuit racing (F-Zero/Wipeout energy; Mars flagship; physics-first ribbon) |

Living notes also: https://hackmd.io/s/HkMqq2btze

## Projects

| Repo | Graphics | Physics / other | Assets | Headless / local | Web |
|------|----------|-----------------|--------|------------------|-----|
| [`vandals-and-heroes`](https://github.com/kvark/vandals-and-heroes) | Blade | Rapier, Winit, Choir | Git LFS | lavapipe | WebGL2 + Pages |
| [`redline`](https://github.com/kvark/redline) | Blade | Rapier joints | Git LFS (Kenney CC0) | lavapipe / Xvfb smoke + scripted drives | WASM + Pages |
| [`geofront`](https://github.com/kvark/geofront) | Blade (pin by **git rev**; see blade#389) | — | Quaternius / LFS-style GLBs; `scripts/fetch-shaders.sh` | `GEOFRONT_AUTOPLAY=1` under Xvfb + lavapipe | WASM |
| [`claymore-blade`](https://github.com/kvark/claymore-blade) | Blade | — (hex hunt, no Rapier) | Kenney CC0 + 2D art (no LFS GLBs) | lavapipe | Pages; see CI note below |
| [`vange-rs`](https://github.com/kvark/vange-rs) | **wgpu** / winit / egui | TCP 7800 / WS 7801 (`vangers-net`) | Public `data-0` release zips (**not** LFS) | Xvfb after `x11-dl` 2.21+ | WASM; clippy/build/WASM Actions |

Two lanes:

1. **Blade family** — redline, geofront, claymore-blade, vandals-and-heroes: Vulkan via lavapipe locally, WebGL2/WASM in browser, usually LFS (or LFS-style) art.
2. **wgpu outlier** — vange-rs: shipped zip game data, multiplayer ports, different smoke story.

## Local defaults (Blade)

- Prefer **lavapipe** (Mesa software Vulkan) for headless / no-GPU iteration.
- Use **Xvfb** when a display is required (`xvfb-run -a …`).
- `git lfs pull` (or project equivalent) before native or WASM builds that embed models.
- Release builds for playable framerate; debug when you need assertions.

## Web / WASM

- Target `wasm32-unknown-unknown`, `wasm-bindgen-cli` version matching `Cargo.lock`.
- WebGL2 browsers; Blade needs GLES-friendly shader linking on wasm (see per-repo `.cargo` / Blade notes).
- Pages deploys from `main` are common; they are **not** a substitute for PR checks.

## CI expectations

| Check | Blade family | vange-rs |
|-------|--------------|----------|
| Native tests / clippy / build | yes | yes |
| Headless smoke (Xvfb ± lavapipe / autoplay) | preferred | Xvfb native smoke |
| **WASM on PRs** | **required** — do not rely only on the Pages job (Claymore today: native + GLES on PR, wasm only on Pages; bring wasm onto PRs) | already on Actions |
| Pages deploy | on `main` where enabled | as configured |

## Open follow-ups

- Land a wasm PR check on Blade repos that still gate web only via Pages (Claymore first).
- Keep Blade pins explicit (git rev) when tracking fixes such as blade#389.
- Extend this table when a new title joins either lane.
