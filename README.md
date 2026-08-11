# Vandals and Heroes

[![Build Status](https://github.com/kvark/vandals-and-heroes/workflows/check/badge.svg)](https://github.com/kvark/vandals-and-heroes/actions)

Prototype game in Vangers universe. Related to [Rusty Vangers](https://kvark.itch.io/vangers).

![v3-sky-and-shadows](/etc/screenshots/v3-sky-and-shadows.jpg)

## Tech stack

- [Blade](https://github.com/kvark/blade) for graphics
- [Rapier](https://github.com/dimforge/rapier) for physics
- [Winit](https://github.com/rust-windowing/winit) for window and events
- [Choir](https://github.com/kvark/choir) for threading

## Instructions

After checking out the repo, make sure you get the LFS artifacts:
```bash
git lfs pull
```
Building is running is just the usual :crab: workflow:
```bash
cargo run
```

## Web

The same game binary runs in the browser on WebGL2. To build it, add the
wasm target and install the `wasm-bindgen` CLI at the version matching
`Cargo.lock` (the CLI and crate versions must agree):

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version $(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | grep version | cut -d'"' -f2)
```

The map, car, and environment assets are embedded into the wasm at compile
time, so `git lfs pull` must have run first. Then:

```bash
cargo build --release --bin game --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/game.wasm --out-dir web/pkg --target web --no-typescript
python3 -m http.server 8080 --directory web
```

and open <http://localhost:8080>. Pushes to `main` deploy the same page to
GitHub Pages automatically (see `.github/workflows/deploy-web.yaml`; Pages
must be enabled with "GitHub Actions" as the source in the repo settings).

## Platforms

Runs on Linux, Android, and Windows with relatiively modern Vulkan driver (old hardware is ok), macOS/iOS, and the Web via WebGL2.
