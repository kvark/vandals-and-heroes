# Vandals and Heroes

[![Build Status](https://github.com/kvark/vandals-and-heroes/workflows/check/badge.svg)](https://github.com/kvark/vandals-and-heroes/actions)

Prototype game in Vangers universe. Related to [Rusty Vangers](https://kvark.itch.io/vangers).

![cylinder map](/etc/screenshots/v1-cylinder-map.jpg)

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

## Platforms

Runs on Linux, Android, and Windows with relatiively modern Vulkan driver (old hardware is ok), and macOS/iOS.
