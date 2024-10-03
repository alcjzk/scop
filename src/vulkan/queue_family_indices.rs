use crate::Error;
use crate::Instance;
use crate::Result;

use ash::vk;

pub type QueueFamilyIndex = u32;

#[derive(Debug)]
pub struct QueueFamilyIndices {
    pub graphics_family: QueueFamilyIndex,
    pub present_family: QueueFamilyIndex,
}

impl QueueFamilyIndices {
    pub unsafe fn builder(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<QueueFamilyIndicesBuilder> {
        QueueFamilyIndicesBuilder::from_device_surface(instance, physical_device, surface)
    }
    pub unsafe fn from_device_surface(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        Self::builder(instance, physical_device, surface)?
            .build()
            .ok_or(Error::Generic("incomplete queue family indices"))
    }
}

#[derive(Default)]
pub struct QueueFamilyIndicesBuilder {
    pub graphics_family: Option<QueueFamilyIndex>,
    pub present_family: Option<QueueFamilyIndex>,
}

impl QueueFamilyIndicesBuilder {
    pub unsafe fn from_device_surface(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let mut indices = Self::default();

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

    pub fn build(self) -> Option<QueueFamilyIndices> {
        Some(QueueFamilyIndices {
            graphics_family: self.graphics_family?,
            present_family: self.present_family?,
        })
    }
}
