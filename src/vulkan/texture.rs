use crate::vulkan;
use crate::IteratorExt;
use crate::Ppm;
use crate::Result;

use ash::vk;
use std::fs::OpenOptions;
use std::path::Path;
use std::ptr;

#[derive(Debug)]
pub struct Texture {
    memory: vk::DeviceMemory,
    image: vk::Image,
    view: vk::ImageView,
}

impl Texture {
    pub unsafe fn new(
        path: impl AsRef<Path>,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;

        let image = Ppm::from_reader(file)?;
        let width = image.width as u32;
        let height = image.height as u32;

        let data = image
            .data
            .chunks_exact(Ppm::COLOR_CHANNEL_COUNT)
            .flat_map(|c| [c[0], c[1], c[2], u8::MAX])
            .collect_vec();

        let size = width * height * 4;

        let staging_buffer_info = vk::BufferCreateInfo::default()
            .size(size as _)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = device.create_buffer(&staging_buffer_info, None)?;

        let staging_buffer_requirements = device.get_buffer_memory_requirements(staging_buffer);

        let memory_type_index = vulkan::find_memory_type(
            instance,
            physical_device,
            staging_buffer_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let staging_buffer_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_buffer_requirements.size)
            .memory_type_index(memory_type_index);

        let staging_buffer_memory = device.allocate_memory(&staging_buffer_alloc_info, None)?;

        device.bind_buffer_memory(staging_buffer, staging_buffer_memory, 0)?;

        {
            let memory_data = device.map_memory(
                staging_buffer_memory,
                0,
                size as _,
                vk::MemoryMapFlags::empty(),
            )?;
            ptr::copy_nonoverlapping(data.as_ptr(), memory_data as *mut u8, size as _);
            device.unmap_memory(staging_buffer_memory);
        }

        let image_extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(image_extent)
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = device.create_image(&image_info, None)?;

        let image_requirements = device.get_image_memory_requirements(image);

        let memory_type_index = vulkan::find_memory_type(
            instance,
            physical_device,
            image_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let image_alloc_info = vk::MemoryAllocateInfo::default()
            .memory_type_index(memory_type_index)
            .allocation_size(image_requirements.size);

        let memory = device.allocate_memory(&image_alloc_info, None)?;

        device.bind_image_memory(image, memory, 0)?;

        vulkan::transition_image_layout(
            device,
            command_pool,
            queue,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;

        vulkan::copy_buffer_to_image(
            device,
            command_pool,
            queue,
            staging_buffer,
            image,
            width,
            height,
        )?;

        vulkan::transition_image_layout(
            device,
            command_pool,
            queue,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        device.free_memory(staging_buffer_memory, None);
        device.destroy_buffer(staging_buffer, None);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let view = device.create_image_view(&view_info, None)?;

        Ok(Self {
            memory,
            image,
            view,
        })
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }
}
