//! Asset access: the filesystem on native, embedded bytes on the web.
//!
//! The web build has no filesystem, and fetching assets over HTTP would push
//! async plumbing through the whole loading path — so the demo's data set is
//! compiled straight into the wasm binary instead. The embedded config
//! (`web_config.ron`) mirrors `data/config.ron` minus the state recorder
//! (nowhere to write a file on the web).

use std::borrow::Cow;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
pub fn read(path: &Path) -> Cow<'static, [u8]> {
    Cow::Owned(
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("Unable to read {}: {e}", path.display())),
    )
}

#[cfg(target_arch = "wasm32")]
pub fn read(path: &Path) -> Cow<'static, [u8]> {
    let key = path
        .to_str()
        .expect("asset path must be utf-8")
        .replace('\\', "/");
    let bytes: &'static [u8] = match key.as_str() {
        "data/config.ron" => include_bytes!("web_config.ron"),
        "data/maps/fostral/map.ron" => include_bytes!("../../data/maps/fostral/map.ron"),
        "data/maps/fostral/map.png" => include_bytes!("../../data/maps/fostral/map.png"),
        "data/envs/Fostral.png" => include_bytes!("../../data/envs/Fostral.png"),
        "data/cars/OxidizeMonk/car.ron" => include_bytes!("../../data/cars/OxidizeMonk/car.ron"),
        "data/cars/OxidizeMonk/body.glb" => include_bytes!("../../data/cars/OxidizeMonk/body.glb"),
        other => panic!("asset {other} is not embedded in the web build"),
    };
    Cow::Borrowed(bytes)
}
