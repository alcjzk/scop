mod command_buffer;
mod instance;
mod queue_family_indices;
mod renderer;
mod surface_object_manager;
mod sync_object_store;
mod texture;

pub use command_buffer::*;
pub use instance::*;
pub use queue_family_indices::*;
pub use renderer::*;
pub use surface_object_manager::*;
pub use sync_object_store::*;
pub use texture::*;

use crate::{Error, Result};
use ash::{self, vk};

pub unsafe fn find_supported_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    formats: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    for format in formats.iter().copied() {
        let properties = instance.get_physical_device_format_properties(physical_device, format);

        match tiling {
            vk::ImageTiling::LINEAR => {
                if properties.linear_tiling_features.contains(features) {
                    return Ok(format);
                }
            }
            vk::ImageTiling::OPTIMAL => {
                if properties.optimal_tiling_features.contains(features) {
                    return Ok(format);
                }
            }
            _ => (),
        }
    }

    Err(Error::Generic("failed to find supported format"))
}

pub unsafe fn find_depth_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<vk::Format> {
    find_supported_format(
        instance,
        physical_device,
        &[
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ],
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

// TODO: Format arg unused? see: depth buffer
pub unsafe fn transition_image_layout(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    _format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let command_buffer = CommandBufferOneTimeSubmit::new(device, &command_pool)?;

    let mut barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    let (source_stage, destination_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
            barrier = barrier
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            )
        }
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
            barrier = barrier
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            )
        }
        _ => panic!("unsupported layout transition"),
    };

    device.cmd_pipeline_barrier(
        *command_buffer,
        source_stage,
        destination_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );
    command_buffer.submit(queue, vk::Fence::null())?;

    Ok(())
}

pub unsafe fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    property_flags: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let properties = instance.get_physical_device_memory_properties(physical_device);

    for memory_type in 0..properties.memory_type_count {
        if type_filter & (1 << memory_type) == 0 {
            continue;
        }

        let memory_type_property_flags =
            properties.memory_types[memory_type as usize].property_flags;

        if memory_type_property_flags & property_flags == property_flags {
            return Ok(memory_type);
        }
    }
    Err(Error::Generic("failed to find suitable memory type"))
}

// TODO: unused?
#[allow(unused)]
pub fn has_stencil_component(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT
    )
}

pub unsafe fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = CommandBufferOneTimeSubmit::new(device, &command_pool)?;

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    device.cmd_copy_buffer_to_image(
        *command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    command_buffer.submit(queue, vk::Fence::null())?;

    Ok(())
}

pub unsafe fn copy_buffer(
    device: &ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = CommandBufferOneTimeSubmit::new(device, &command_pool)?;

    let copy_regions = &[vk::BufferCopy::default().size(size)];
    device.cmd_copy_buffer(*command_buffer, source, destination, copy_regions);

    command_buffer.submit(queue, vk::Fence::null())?;

    Ok(())
}
