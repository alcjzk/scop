use ash::khr::swapchain;
use std::{fmt, ops};

pub struct Device {
    inner: ash::Device,
    swapchain_khr: swapchain::Device,
}

impl Device {
    pub unsafe fn new(instance: &ash::Instance, inner: ash::Device) -> Self {
        let swapchain_khr = swapchain::Device::new(instance, &inner);
        Self {
            inner,
            swapchain_khr,
        }
    }
    pub fn swapchain_khr(&self) -> &swapchain::Device {
        &self.swapchain_khr
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device").finish()
    }
}

impl ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ops::DerefMut for Device {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.inner.destroy_device(None);
        }
    }
}
