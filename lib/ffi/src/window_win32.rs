use std::ffi::c_void;

use raw_window_handle as rwh;

#[repr(C)]
pub struct VangersWin32Window {
    pub hwnd: *mut c_void,
    pub hinstance: *mut c_void,
}

impl rwh::HasWindowHandle for VangersWin32Window {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        let hwnd = std::num::NonZeroIsize::new(self.hwnd as isize).unwrap();
        let hinstance = std::num::NonZeroIsize::new(self.hinstance as isize);
        let mut handle = rwh::Win32WindowHandle::new(hwnd);
        handle.hinstance = hinstance;
        unsafe {
            Ok(rwh::WindowHandle::borrow_raw(rwh::RawWindowHandle::Win32(
                handle,
            )))
        }
    }
}
impl rwh::HasDisplayHandle for VangersWin32Window {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        Ok(rwh::DisplayHandle::windows())
    }
}
