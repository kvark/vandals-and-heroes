use std::ffi::c_void;
use std::ptr;
use env_logger;
use blade_graphics as gpu;
use vandals_and_heroes::{
    camera::Camera,
    render::Render,

};

mod window_win32;
mod context;

use context::Context;
use window_win32::VangersWin32Window;


#[no_mangle]
pub extern "C" fn vangers_init(hwnd: *mut c_void, hinstance: *mut c_void, width: u32, height: u32) -> Option<ptr::NonNull<Context>> {
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
        },
        None => None
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