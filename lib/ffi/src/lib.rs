extern crate core;

use blade_graphics as gpu;
use env_logger;
use std::ffi::c_void;
use std::{ptr, slice};

mod context;
mod window_win32;

use context::Context;
use window_win32::VangersWin32Window;

#[no_mangle]
pub extern "C" fn vangers_init(
    hwnd: *mut c_void,
    hinstance: *mut c_void,
    width: u32,
    height: u32,
) -> Option<ptr::NonNull<Context>> {
    let _ = env_logger::try_init();

    let extent = gpu::Extent {
        width,
        height,
        depth: 1,
    };

    let handle = VangersWin32Window { hwnd, hinstance };
    match Context::new(extent, &handle) {
        Some(context) => {
            let ptr = Box::into_raw(Box::new(context));
            ptr::NonNull::new(ptr)
        }
        None => None,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vangers_exit(ctx: *mut Context) {
    let _ctx = Box::from_raw(ctx);
}

#[no_mangle]
pub unsafe extern "C" fn vangers_resize(ctx: &mut Context, width: u32, height: u32) {
    ctx.resize(width, height);
}

#[no_mangle]
pub unsafe extern "C" fn vangers_redraw(ctx: &mut Context) {
    ctx.render()
}

#[no_mangle]
unsafe extern "C" fn vangers_set_map(
    ctx: &mut Context,
    radius_min: f32,
    radius_max: f32,
    width: u32,
    height: u32,
    pbuf: *const u8,
) {
    log::info!(
        "vangers_set_map rmin: {}, rmax: {}, width: {}, height: {}",
        radius_min,
        radius_max,
        width,
        height
    );
    let map_config = vandals_and_heroes::config::Map {
        radius: core::ops::Range {
            start: radius_min,
            end: radius_max,
        },
    };
    let len = width * height * 4;
    let buf = slice::from_raw_parts(pbuf, len as usize);
    ctx.set_map(map_config, width, height, buf);
}
