use crate::Instance;
use crate::Result;

use ash::vk;

pub type QueueFamilyIndex = u32;

#[derive(Debug, Default)]
pub struct QueueFamilyIndices {
    pub graphics_family: Option<QueueFamilyIndex>,
    pub present_family: Option<QueueFamilyIndex>,
}

impl QueueFamilyIndices {
    pub unsafe fn from_device_surface(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let mut indices = QueueFamilyIndices::default();
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        for (idx, family) in (0u32..).zip(properties.iter()) {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(idx);
            }
            if instance.surface_khr().get_physical_device_surface_support(
                physical_device,
                idx,
                surface,
            )? {
                indices.present_family = Some(idx);
            }
            if indices.is_complete() {
                break;
            }
        }
        Ok(indices)
    }

    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}
