use crate::util::IteratorExt;
use crate::vulkan;
use crate::Instance;
use crate::QueueFamilyIndex;
use crate::QueueFamilyIndices;
use crate::Vertex;
use crate::{Error, Result, SurfaceObjectManager, SyncObjectStore, Texture, UniformBufferObject};
use crate::{Matrix, SurfaceSupportDetails};

use ash::vk;
use ash::Entry as Vulkan;
use glfw::{Glfw, PWindow};

use std::collections::HashSet;
use std::ffi::{c_char, c_uint, CStr};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr;
use std::ptr::NonNull;

pub struct MappedArray<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> MappedArray<T> {
    unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len }
    }
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }
}

pub struct Renderer {
    _vulkan: Vulkan,
    instance: Instance,
    surface: vk::SurfaceKHR,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    sync_objects: SyncObjectStore,
    ubo_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_mapped: MappedArray<UniformBufferObject>,
    sampler: vk::Sampler,
    surface_objects: SurfaceObjectManager,
    vertex_memory: vk::DeviceMemory,
    vertex_buffer: vk::Buffer,
    index_count: u32,
    index_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    texture: ManuallyDrop<Texture>,
    frame_index: u32,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Renderer {
    const INSTANCE_EXTENSIONS: &[&CStr] = &[
        vk::KHR_PORTABILITY_ENUMERATION_NAME,
        vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME,
    ];

    const DEVICE_EXTENSIONS: &[&CStr] = &[
        vk::KHR_SWAPCHAIN_NAME,
        #[cfg(target_os = "macos")] // TODO:
        vk::KHR_PORTABILITY_SUBSET_NAME,
    ];

    const VALIDATION_LAYERS: &[&CStr] = &[
        #[cfg(debug_assertions)]
        c"VK_LAYER_KHRONOS_validation",
    ];

    const MAX_FRAMES_IN_FLIGHT: u32 = 2;

    pub fn new(
        glfw: &Glfw,
        window: &PWindow,
        vertices: Vec<Vertex<f32>>,
        indices: Vec<u16>,
        texture_path: impl AsRef<Path>,
    ) -> Result<Self> {
        unsafe {
            let vulkan = Vulkan::load()?;
            let instance = create_instance(glfw, &vulkan)?;
            let surface = create_window_surface(window, &instance)?;

            let physical_device =
                select_physical_device(&instance, surface, Self::DEVICE_EXTENSIONS)?;

            let (device, graphics_queue, present_queue) = create_logical_device(
                &instance,
                physical_device,
                surface,
                Self::DEVICE_EXTENSIONS,
            )?;

            // TODO: fetched repetitively
            let queue_families =
                QueueFamilyIndices::from_device_surface(&instance, physical_device, surface)?;

            let command_pool =
                create_command_pool(&device, queue_families.graphics_family.unwrap())?;
            let command_buffers =
                create_command_buffers(&device, command_pool, Self::MAX_FRAMES_IN_FLIGHT)?;

            let descriptor_pool = create_descriptor_pool(&device, Self::MAX_FRAMES_IN_FLIGHT)?;
            let descriptor_set_layout = create_descriptor_set_layout(&device)?;

            let sync_objects = SyncObjectStore::new(&device, Self::MAX_FRAMES_IN_FLIGHT as _)?;

            let (ubo_memory, uniform_buffers, uniform_buffers_mapped) = create_uniform_buffers(
                &instance,
                physical_device,
                &device,
                Self::MAX_FRAMES_IN_FLIGHT as _,
            )?;

            let sampler = create_texture_sampler(&instance, physical_device, &device)?;

            let (width, height) = window.get_framebuffer_size();

            let target_extent = vk::Extent2D {
                width: width as _,
                height: height as _,
            };

            let surface_objects = SurfaceObjectManager::new(
                &instance,
                physical_device,
                &device,
                surface,
                target_extent,
                descriptor_set_layout,
            )?;

            let (vertex_memory, vertex_buffer) = create_buffer_device_local(
                &instance,
                physical_device,
                &device,
                graphics_queue,
                command_pool,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &vertices,
            )?;

            let (index_memory, index_buffer) = create_buffer_device_local(
                &instance,
                physical_device,
                &device,
                graphics_queue,
                command_pool,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &indices,
            )?;

            let texture = ManuallyDrop::new(Texture::new(
                texture_path,
                &instance,
                physical_device,
                &device,
                command_pool,
                graphics_queue,
            )?);

            let descriptor_sets = create_descriptor_sets(
                &device,
                descriptor_set_layout,
                descriptor_pool,
                &uniform_buffers,
                texture.view(),
                sampler,
            )?;

            Ok(Self {
                _vulkan: vulkan,
                instance,
                surface,
                device,
                graphics_queue,
                present_queue,
                command_pool,
                command_buffers,
                descriptor_pool,
                descriptor_set_layout,
                sync_objects,
                ubo_memory,
                uniform_buffers,
                uniform_buffers_mapped,
                sampler,
                surface_objects,
                vertex_memory,
                vertex_buffer,
                index_count: indices.len() as _,
                index_memory,
                index_buffer,
                texture,
                frame_index: 0,
                descriptor_sets,
            })
        }
    }

    pub unsafe fn draw_frame(&mut self) -> Result<bool> {
        let swapchain_device = ash::khr::swapchain::Device::new(&self.instance, &self.device);

        let in_flight = self.sync_objects.get_in_flight(self.frame_index);
        let image_available = self.sync_objects.get_image_available(self.frame_index);
        let render_finished = self.sync_objects.get_render_finished(self.frame_index);
        let command_buffer = self.command_buffers[self.frame_index as usize];

        self.device.wait_for_fences(&[in_flight], true, u64::MAX)?;

        let image_index = match swapchain_device.acquire_next_image(
            self.swapchain(),
            u64::MAX,
            image_available,
            vk::Fence::null(),
        ) {
            Ok((image_index, _is_surface_suboptimal)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Ok(true);
            }
            Err(error) => return Err(error.into()),
        };

        self.device.reset_fences(&[in_flight])?;

        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        self.record_command_buffer(command_buffer, image_index)?;

        let vk::Extent2D { width, height } = self.image_extent();
        let aspect_ratio = width as f32 / height as f32;

        let mut ubo = UniformBufferObject {
            model: Matrix::identity().rotate(-20.0f32.to_radians(), [0.0, 0.0, 1.0]),
            view: Matrix::look_at([2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            projection: Matrix::perspective(45.0f32.to_radians(), aspect_ratio, 0.1, 10.0),
        };
        ubo.projection[1][1] *= -1.0; // TODO: Do elsewhere?

        self.uniform_buffers_mapped.as_mut_slice()[self.frame_index as usize] = ubo;

        let mut submit_info = vk::SubmitInfo {
            p_wait_dst_stage_mask: [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT].as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[self.frame_index as usize],
            ..Default::default()
        };

        let wait_semaphores = &[image_available];
        let signal_semaphores = &[render_finished];

        submit_info = submit_info.wait_semaphores(wait_semaphores);
        submit_info = submit_info.signal_semaphores(signal_semaphores);

        self.device
            .queue_submit(self.graphics_queue, &[submit_info], in_flight)?;

        let mut present_info = vk::PresentInfoKHR {
            swapchain_count: 1,
            p_swapchains: &self.swapchain(),
            p_image_indices: &image_index,
            ..Default::default()
        };

        present_info = present_info.wait_semaphores(signal_semaphores);

        self.frame_index = (self.frame_index + 1) % Self::MAX_FRAMES_IN_FLIGHT;

        let result = swapchain_device.queue_present(self.present_queue, &present_info);

        match result {
            Ok(false) => Ok(false),
            Err(error) if error != vk::Result::ERROR_OUT_OF_DATE_KHR => Err(error.into()),
            _ => Ok(true),
        }
    }

    unsafe fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
    ) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        self.device
            .begin_command_buffer(command_buffer, &begin_info)?;

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0f32, 0.0f32, 0.0f32, 1.0f32],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clear_values = &[clear_color, clear_depth];
        let extent = self.image_extent();

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.surface_objects.render_pass())
            .framebuffer(self.surface_objects.get_framebuffer(image_index))
            .clear_values(clear_values)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            });

        let viewport = vk::Viewport {
            x: 0.0f32,
            y: 0.0f32,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0f32,
            max_depth: 1.0f32,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        self.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.surface_objects.graphics_pipeline(),
        );

        self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

        let vertex_buffers = [self.vertex_buffer];
        let offsets = [0u64];

        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.index_buffer,
            0,
            vk::IndexType::UINT16,
        );

        let sets = &[self.descriptor_sets[self.frame_index as usize]];
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.surface_objects.pipeline_layout(),
            0,
            sets,
            &[],
        );
        self.device
            .cmd_draw_indexed(command_buffer, self.index_count, 1, 0, 0, 0);

        self.device.cmd_end_render_pass(command_buffer);
        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    pub unsafe fn resize(&mut self, target_extent: vk::Extent2D) -> Result<()> {
        self.surface_objects
            .resize(&self.instance, &self.device, target_extent)
    }

    fn swapchain(&self) -> vk::SwapchainKHR {
        self.surface_objects.swapchain()
    }

    fn image_extent(&self) -> vk::Extent2D {
        self.surface_objects.image_extent()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.texture.destroy(&self.device);
            ManuallyDrop::drop(&mut self.texture);
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_memory, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_memory, None);
            self.surface_objects.destroy(&self.instance, &self.device);
            self.device.destroy_sampler(self.sampler, None);
            for buffer in std::mem::take(&mut self.uniform_buffers) {
                self.device.destroy_buffer(buffer, None);
            }
            self.device.free_memory(self.ubo_memory, None);
            self.sync_objects.destroy(&self.device);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            self.device.destroy_command_pool(self.command_pool, None);

            //self.device.device_wait_idle();
            self.device.destroy_device(None);

            self.instance
                .surface_khr()
                .destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn create_instance(glfw: &Glfw, vulkan: &Vulkan) -> Result<Instance> {
    let api_version = vk::make_api_version(0, 1, 0, 0);
    let application_info = vk::ApplicationInfo::default().api_version(api_version);

    let mut required_extensions = glfw_required_instance_extensions(glfw)?;
    required_extensions.extend(Renderer::INSTANCE_EXTENSIONS);

    let required_extensions = required_extensions
        .into_iter()
        .map(|name| name.as_ptr())
        .collect_vec();

    let enabled_layers = Renderer::VALIDATION_LAYERS
        .iter()
        .map(|name| name.as_ptr())
        .collect_vec();

    let layer_properties = vulkan.enumerate_instance_layer_properties()?;

    for layer_name in Renderer::VALIDATION_LAYERS.iter().copied() {
        if !layer_properties
            .iter()
            .map(|p| p.layer_name_as_c_str().expect("expected valid c str"))
            .any(|name| name.eq(layer_name))
        {
            return Err(Error::Generic("missing layer"));
        }
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_extension_names(&required_extensions)
        .enabled_layer_names(&enabled_layers)
        .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

    let inner = vulkan.create_instance(&create_info, None)?;

    Ok(Instance::new(vulkan, inner))
}

fn create_window_surface(window: &PWindow, instance: &Instance) -> Result<vk::SurfaceKHR> {
    let mut surface = vk::SurfaceKHR::null();

    window
        .create_window_surface(instance.handle(), std::ptr::null(), &mut surface)
        .result()?;

    Ok(surface)
}

unsafe fn glfw_required_instance_extensions(_glfw: &Glfw) -> Result<HashSet<&CStr>> {
    use glfw::ffi::glfwGetRequiredInstanceExtensions;

    let mut count: c_uint = 0;

    let required_extensions_ptr = glfwGetRequiredInstanceExtensions(&mut count);
    let mut required_extensions_ptr = NonNull::new(required_extensions_ptr as *mut *const c_char)
        .ok_or(Error::Generic(
        "failed to get glfw required instance extensions",
    ))?;

    if count == 0 {
        required_extensions_ptr = NonNull::dangling();
    }

    let required_extensions =
        std::slice::from_raw_parts(required_extensions_ptr.as_ptr(), count as _);

    let required_extensions = required_extensions
        .iter()
        .copied()
        .map(|name| {
            debug_assert!(!name.is_null());
            CStr::from_ptr(name)
        })
        .collect_set();

    Ok(required_extensions)
}

unsafe fn select_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
    required_extensions: &[&CStr],
) -> Result<vk::PhysicalDevice> {
    let devices = instance.enumerate_physical_devices()?;

    // TODO: Device rating?

    devices
        .into_iter()
        .try_find(|device| is_device_suitable(instance, *device, surface, required_extensions))?
        .ok_or(Error::Generic("no suitable device found"))
}

unsafe fn is_device_extensions_supported(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    required_extensions: &[&CStr],
) -> Result<bool> {
    let available_extensions = instance.enumerate_device_extension_properties(physical_device)?;
    let mut required_extensions = required_extensions.iter().copied().collect_set();

    for extension in available_extensions {
        let _ = required_extensions.remove(extension.extension_name_as_c_str()?);
    }

    Ok(required_extensions.is_empty())
}

unsafe fn is_device_suitable(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    required_extensions: &[&CStr],
) -> Result<bool> {
    let features = instance.get_physical_device_features(physical_device);
    let queue_family_indices =
        QueueFamilyIndices::from_device_surface(instance, physical_device, surface)?;

    if !queue_family_indices.is_complete() {
        return Ok(false);
    }

    if !is_device_extensions_supported(instance, physical_device, required_extensions)? {
        return Ok(false);
    }

    let surface_support =
        SurfaceSupportDetails::from_device_surface(instance, physical_device, surface)?;

    if surface_support.formats.is_empty() {
        return Ok(false);
    }
    if surface_support.present_modes.is_empty() {
        return Ok(false);
    }
    if features.sampler_anisotropy == vk::FALSE {
        return Ok(false);
    }

    Ok(true)
}

unsafe fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    extensions: &[&CStr],
) -> Result<(ash::Device, vk::Queue, vk::Queue)> {
    let queue_family_indices =
        QueueFamilyIndices::from_device_surface(instance, physical_device, surface)?;

    let graphics_family = queue_family_indices.graphics_family.unwrap();
    let present_family = queue_family_indices.present_family.unwrap();
    let unique_queue_families = [graphics_family, present_family].into_iter().collect_set();

    let queue_priorities = &[1.0];

    let queue_create_infos = unique_queue_families
        .into_iter()
        .map(|index| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(index)
                .queue_priorities(queue_priorities)
        })
        .collect_vec();

    let enabled_extension_names = extensions
        .iter()
        .copied()
        .map(|name| name.as_ptr())
        .collect_vec();

    let enabled_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

    let device_create_info = vk::DeviceCreateInfo::default()
        .enabled_features(&enabled_features)
        .enabled_extension_names(&enabled_extension_names)
        .queue_create_infos(&queue_create_infos);

    let device = instance.create_device(physical_device, &device_create_info, None)?;

    let graphics_queue = device.get_device_queue(graphics_family, 0);
    let present_queue = device.get_device_queue(present_family, 0);

    Ok((device, graphics_queue, present_queue))
}

unsafe fn create_command_pool(
    device: &ash::Device,
    queue_family_index: QueueFamilyIndex,
) -> Result<vk::CommandPool> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);

    Ok(device.create_command_pool(&create_info, None)?)
}

unsafe fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    max_frames_in_flight: u32,
) -> Result<Vec<vk::CommandBuffer>> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(max_frames_in_flight);

    Ok(device.allocate_command_buffers(&alloc_info)?)
}

unsafe fn create_descriptor_pool(
    device: &ash::Device,
    max_frames_in_flight: u32,
) -> Result<vk::DescriptorPool> {
    let ubo_pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(max_frames_in_flight);

    let sampler_pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(max_frames_in_flight);

    let pool_sizes = &[ubo_pool_size, sampler_pool_size];

    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(pool_sizes)
        .max_sets(max_frames_in_flight);

    Ok(device.create_descriptor_pool(&create_info, None)?)
}

unsafe fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let sampler_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        // TODO:: ?
        //.immutable_samplers(&[])
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = &[ubo_binding, sampler_binding];

    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

    Ok(device.create_descriptor_set_layout(&create_info, None)?)
}

unsafe fn create_uniform_buffers(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    frame_count: usize,
) -> Result<(
    vk::DeviceMemory,
    Vec<vk::Buffer>,
    MappedArray<UniformBufferObject>,
)> {
    let buffer_size = std::mem::size_of::<UniformBufferObject>();

    let buffer_info = vk::BufferCreateInfo::default()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .size(buffer_size as _);

    let buffers = std::iter::repeat(buffer_info)
        .take(frame_count)
        .map(|info| Ok(device.create_buffer(&info, None)?))
        .try_collect_vec()?;

    let memory_requirements = device.get_buffer_memory_requirements(buffers[0]);

    let memory_property_flags =
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE;

    let memory_type_idx = vulkan::find_memory_type(
        instance,
        physical_device,
        memory_requirements.memory_type_bits,
        memory_property_flags,
    )?;

    let allocation_size = memory_requirements.size * frame_count as u64;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(allocation_size)
        .memory_type_index(memory_type_idx);

    let memory = device.allocate_memory(&alloc_info, None)?;

    for (idx, buffer) in buffers.iter().copied().enumerate() {
        let offset = idx as u64 * memory_requirements.size;
        device.bind_buffer_memory(buffer, memory, offset)?;
    }

    let data = device.map_memory(memory, 0, allocation_size, vk::MemoryMapFlags::empty())?;
    let mapped = MappedArray::from_raw_parts(data as _, frame_count);

    Ok((memory, buffers, mapped))
}

unsafe fn create_texture_sampler(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
) -> Result<vk::Sampler> {
    let physical_device_properties = instance.get_physical_device_properties(physical_device);

    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(physical_device_properties.limits.max_sampler_anisotropy)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);

    Ok(device.create_sampler(&create_info, None)?)
}

unsafe fn create_buffer_device_local<V>(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    buffer_usage: vk::BufferUsageFlags,
    contents: &[V],
) -> Result<(vk::DeviceMemory, vk::Buffer)> {
    let buffer_size = size_of_val(contents);

    let staging_buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size as _)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = device.create_buffer(&staging_buffer_info, None)?;
    let staging_memory_requirements = device.get_buffer_memory_requirements(staging_buffer);

    let staging_memory_properties =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let staging_memory_type = vulkan::find_memory_type(
        instance,
        physical_device,
        staging_memory_requirements.memory_type_bits,
        staging_memory_properties,
    )?;

    let staging_alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(staging_memory_requirements.size)
        .memory_type_index(staging_memory_type);

    let staging_memory = device.allocate_memory(&staging_alloc_info, None)?;

    device.bind_buffer_memory(staging_buffer, staging_memory, 0)?;

    {
        let staging_data = device.map_memory(
            staging_memory,
            0,
            buffer_size as _,
            vk::MemoryMapFlags::empty(),
        )?;
        ptr::copy_nonoverlapping(contents.as_ptr(), staging_data as _, contents.len());
    }
    device.unmap_memory(staging_memory);

    let vertex_buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size as _)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | buffer_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let vertex_buffer = device.create_buffer(&vertex_buffer_info, None)?;
    let vertex_memory_requirements = device.get_buffer_memory_requirements(vertex_buffer);

    let vertex_memory_properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;

    let vertex_memory_type = vulkan::find_memory_type(
        instance,
        physical_device,
        vertex_memory_requirements.memory_type_bits,
        vertex_memory_properties,
    )?;

    let vertex_alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(vertex_memory_requirements.size)
        .memory_type_index(vertex_memory_type);

    let vertex_memory = device.allocate_memory(&vertex_alloc_info, None)?;

    device.bind_buffer_memory(vertex_buffer, vertex_memory, 0)?;

    vulkan::copy_buffer(
        device,
        queue,
        command_pool,
        staging_buffer,
        vertex_buffer,
        buffer_size as _,
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);

    Ok((vertex_memory, vertex_buffer))
}

unsafe fn create_descriptor_sets(
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    uniform_buffers: &[vk::Buffer],
    texture_view: vk::ImageView,
    texture_sampler: vk::Sampler,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = std::iter::repeat(layout)
        .take(uniform_buffers.len())
        .collect_vec();

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    let descriptor_sets = device.allocate_descriptor_sets(&alloc_info)?;

    for (uniform_buffer, descriptor_set) in uniform_buffers
        .iter()
        .copied()
        .zip(descriptor_sets.iter().copied())
    {
        let buffer_infos = &[vk::DescriptorBufferInfo::default()
            .offset(0)
            .buffer(uniform_buffer)
            .range(vk::WHOLE_SIZE)];

        let image_infos = &[vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture_view)
            .sampler(texture_sampler)];

        let descriptor_writes = &[
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_infos),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_infos),
        ];

        device.update_descriptor_sets(descriptor_writes, &[]);
    }

    Ok(descriptor_sets)
}
