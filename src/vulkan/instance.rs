use ash::khr::surface;
use std::fmt;

pub struct Instance {
    inner: ash::Instance,
    surface_khr: surface::Instance,
}

impl Instance {
    pub fn new(vulkan: &ash::Entry, inner: ash::Instance) -> Self {
        let surface_khr = surface::Instance::new(vulkan, &inner);
        Self { inner, surface_khr }
    }

    pub fn surface_khr(&self) -> &surface::Instance {
        &self.surface_khr
    }
}

impl std::ops::Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanInstance").finish()
    }
}
