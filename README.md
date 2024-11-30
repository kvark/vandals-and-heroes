# Vandals and Heroes

[![Build Status](https://github.com/kvark/vandals-and-heroes/workflows/check/badge.svg)](https://github.com/kvark/vandals-and-heroes/actions)

Prototype game in Vangers universe. Related to [Rusty Vangers](https://kvark.itch.io/vangers).

## Tech stack

- [Blade](https://github.com/kvark/blade) for graphics (GPU access)
- [winit](https://github.com/rust-windowing/winit) for window and events
- [choir](https://github.com/kvark/choir) for threading

## Instructions

Just the usual :crab: workflow:
```bash
cargo run
```

## Platforms

Runs on Linux, Android, and Windows with relatiively modern Vulkan driver (old hardware is ok), and macOS/iOS.
