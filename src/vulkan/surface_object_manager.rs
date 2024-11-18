use crate::util::IteratorExt;
use crate::vertex::Vertex;
use crate::vulkan;
use crate::Device;
use crate::QueueFamilyIndices;
use crate::{Error, Instance, Result};

use ash::vk;

pub const FRAGMENT_SHADER: &Align4<[u8]> = &Align4(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/shader.frag.spv"
)));

pub const VERTEX_SHADER: &Align4<[u8]> = &Align4(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/shader.vert.spv"
)));

#[must_use]
#[derive(Debug, Default)]
pub struct SurfaceSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceSupportDetails {
    pub unsafe fn from_device_surface(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        Ok(Self {
            formats: instance
                .surface_khr()
                .get_physical_device_surface_formats(physical_device, surface)?,
            capabilities: instance
                .surface_khr()
                .get_physical_device_surface_capabilities(physical_device, surface)?,
            present_modes: instance
                .surface_khr()
                .get_physical_device_surface_present_modes(physical_device, surface)?,
        })
    }
}

#[must_use]
#[derive(Debug)]
struct SizedObjects {
    extent: vk::Extent2D,
    depth_buffer_memory: vk::DeviceMemory,
    depth_buffer_image: vk::Image,
    depth_buffer_view: vk::ImageView,
    swapchain: vk::SwapchainKHR,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
}

impl SizedObjects {
    unsafe fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        surface: vk::SurfaceKHR,
        surface_capabilities: &vk::SurfaceCapabilitiesKHR,
        surface_format: vk::SurfaceFormatKHR,
        target_extent: vk::Extent2D,
        present_mode: vk::PresentModeKHR,
        render_pass: vk::RenderPass,
    ) -> Result<Self> {
        let extent = select_extent(target_extent, surface_capabilities);

        let (depth_buffer_memory, depth_buffer_image, depth_buffer_view) =
            create_depth_buffer(instance, physical_device, device, extent)?;

        let (swapchain, images) = create_swap_chain(
            instance,
            physical_device,
            device,
            surface,
            surface_capabilities,
            surface_format,
            extent,
            present_mode,
        )?;

        let image_views = create_image_views(device, &images, surface_format.format)?;

        let framebuffers =
            create_framebuffers(device, render_pass, extent, depth_buffer_view, &image_views)?;

        Ok(Self {
            extent,
            depth_buffer_memory,
            depth_buffer_image,
            depth_buffer_view,
            swapchain,
            image_views,
            framebuffers,
        })
    }

    unsafe fn destroy(&mut self, device: &Device) {
        device.device_wait_idle().unwrap();
        self.framebuffers
            .iter()
            .copied()
            .for_each(|f| device.destroy_framebuffer(f, None));

        self.image_views
            .iter()
            .copied()
            .for_each(|v| device.destroy_image_view(v, None));

        device
            .swapchain_khr()
            .destroy_swapchain(self.swapchain, None);

        device.destroy_image_view(self.depth_buffer_view, None);
        device.destroy_image(self.depth_buffer_image, None);
        device.free_memory(self.depth_buffer_memory, None);
    }
}

/// Manager for surface format / extent dependent objects.
#[must_use]
#[derive(Debug)]
pub struct SurfaceObjectManager {
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
    surface_format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    sized: SizedObjects,
}

impl SurfaceObjectManager {
    /// This format will be selected if available.
    pub const PREFERRED_FORMAT: vk::SurfaceFormatKHR = vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };

    pub const PREFERRED_PRESENT_MODE: vk::PresentModeKHR = vk::PresentModeKHR::MAILBOX;

    pub unsafe fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        surface: vk::SurfaceKHR,
        target_extent: vk::Extent2D,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        let surface_support =
            SurfaceSupportDetails::from_device_surface(instance, physical_device, surface)?;

        let surface_format =
            select_surface_format(&surface_support.formats, Self::PREFERRED_FORMAT)
                .ok_or(Error::Generic("could not select a surface format"))?;
        let render_pass =
            create_render_pass(instance, physical_device, device, surface_format.format)?;
        let (pipeline_layout, graphics_pipeline) =
            create_graphics_pipeline(device, descriptor_set_layout, render_pass)?;

        let present_mode =
            select_present_mode(&surface_support.present_modes, Self::PREFERRED_PRESENT_MODE);

        let sized = SizedObjects::new(
            instance,
            physical_device,
            device,
            surface,
            &surface_support.capabilities,
            surface_format,
            target_extent,
            present_mode,
            render_pass,
        )?;

        Ok(Self {
            physical_device,
            surface,
            surface_capabilities: surface_support.capabilities,
            surface_format,
            present_mode,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            sized,
        })
    }

    // TODO:
    #[deprecated(since = "TBD", note = "to be replaced with RAII")]
    pub unsafe fn destroy(&mut self, device: &Device) {
        self.sized.destroy(device);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_pipeline(self.graphics_pipeline, None);
        device.destroy_render_pass(self.render_pass, None);
    }

    pub unsafe fn resize(
        &mut self,
        instance: &Instance,
        device: &Device,
        target_extent: vk::Extent2D,
    ) -> Result<()> {
        self.sized.destroy(device);
        self.sized = SizedObjects::new(
            instance,
            self.physical_device,
            device,
            self.surface,
            &self.surface_capabilities,
            self.surface_format,
            target_extent,
            self.present_mode,
            self.render_pass,
        )?;

        Ok(())
    }

    pub fn swapchain(&self) -> vk::SwapchainKHR {
        self.sized.swapchain
    }

    pub fn image_extent(&self) -> vk::Extent2D {
        self.sized.extent
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn graphics_pipeline(&self) -> vk::Pipeline {
        self.graphics_pipeline
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn get_framebuffer(&self, image_index: u32) -> vk::Framebuffer {
        self.sized.framebuffers[image_index as usize]
    }
}

fn select_surface_format(
    available_formats: &[vk::SurfaceFormatKHR],
    preferred_format: vk::SurfaceFormatKHR,
) -> Option<vk::SurfaceFormatKHR> {
    available_formats
        .iter()
        .copied()
        .find(|f| f == &preferred_format)
        .or_else(|| available_formats.first().copied())
}

unsafe fn create_render_pass(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    image_format: vk::Format,
) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(image_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachments = &[vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    // TODO: Duplicated fetch of format?
    let depth_format = vulkan::find_depth_format(instance, physical_device)?;

    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let depth_attachment_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpasses = &[vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_attachment_ref)];

    let dependencies = &[vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )];

    let attachments = &[color_attachment, depth_attachment];

    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    let render_pass = device.create_render_pass(&create_info, None)?;

    Ok(render_pass)
}

unsafe fn create_graphics_pipeline(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    render_pass: vk::RenderPass,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vertex_shader = create_shader_module(device, VERTEX_SHADER)?;
    let fragment_shader = create_shader_module(device, FRAGMENT_SHADER)?;

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(c"main"),
    ];

    let vertex_attribute_descriptions = [
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: std::mem::offset_of!(Vertex<f32>, position) as _,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: std::mem::offset_of!(Vertex<f32>, color) as _,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 2,
            format: vk::Format::R32G32_SFLOAT,
            offset: std::mem::offset_of!(Vertex<f32>, texture_position) as _,
        },
    ];

    let vertex_binding_descriptions = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex<f32>>() as _,
        input_rate: vk::VertexInputRate::VERTEX,
    }];

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&vertex_attribute_descriptions)
        .vertex_binding_descriptions(&vertex_binding_descriptions);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissor_count(1)
        .viewport_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false)];

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);

    let set_layouts = [descriptor_set_layout];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .depth_stencil_state(&depth_stencil)
        .stages(&shader_stages);

    let pipelines = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        .expect("failed to create graphics pipeline");

    device.destroy_shader_module(vertex_shader, None);
    device.destroy_shader_module(fragment_shader, None);

    debug_assert_eq!(pipelines.len(), 1);

    Ok((pipeline_layout, pipelines[0]))
}

unsafe fn create_shader_module(
    device: &ash::Device,
    bytecode: &Align4<[u8]>,
) -> Result<vk::ShaderModule> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(bytecode.as_u32_slice());
    let module = device.create_shader_module(&create_info, None)?;

    Ok(module)
}

#[repr(align(4))]
pub struct Align4<T: ?Sized>(pub T);

impl Align4<[u8]> {
    pub fn as_u32_slice(&self) -> &[u32] {
        unsafe {
            let (before, slice, after) = self.0.align_to::<u32>();
            debug_assert!(before.is_empty());
            if !after.is_empty() {
                panic!("invalid shader bytecode length");
            }
            slice
        }
    }
}

fn select_present_mode(
    available_modes: &[vk::PresentModeKHR],
    preferred_mode: vk::PresentModeKHR,
) -> vk::PresentModeKHR {
    available_modes
        .iter()
        .copied()
        .find(|m| *m == preferred_mode)
        .unwrap_or_else(|| {
            debug_assert!(available_modes.contains(&vk::PresentModeKHR::FIFO));
            vk::PresentModeKHR::FIFO
        })
}

fn select_extent(
    target_extent: vk::Extent2D,
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }

    let vk::Extent2D { width, height } = target_extent;

    vk::Extent2D {
        width: width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: height.clamp(
            capabilities.max_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

unsafe fn create_depth_buffer(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    extent: vk::Extent2D,
) -> Result<(vk::DeviceMemory, vk::Image, vk::ImageView)> {
    let vk::Extent2D { width, height } = extent;

    let format = vulkan::find_depth_format(instance, physical_device)?;

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .mip_levels(1)
        .array_layers(1)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .format(format);

    let image = device.create_image(&image_info, None)?;

    let memory_requirements = device.get_image_memory_requirements(image);

    let memory_type_index = vulkan::find_memory_type(
        instance,
        physical_device,
        memory_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = device.allocate_memory(&allocate_info, None)?;

    device.bind_image_memory(image, memory, 0)?;

    let image_view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .base_array_layer(0)
                .level_count(1)
                .layer_count(1),
        );

    let image_view = device.create_image_view(&image_view_info, None)?;

    Ok((memory, image, image_view))
}

unsafe fn create_swap_chain(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    surface: vk::SurfaceKHR,
    surface_capabilities: &vk::SurfaceCapabilitiesKHR,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>)> {
    let mut image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0 {
        image_count = image_count.min(surface_capabilities.max_image_count);
    }

    let indices = QueueFamilyIndices::from_device_surface(instance, physical_device, surface)?;
    let queue_family_indices = vec![indices.graphics_family, indices.present_family];

    let mut create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);

    if indices.graphics_family != indices.present_family {
        create_info = create_info
            .image_sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&queue_family_indices);
    }

    let swapchain = device
        .swapchain_khr()
        .create_swapchain(&create_info, None)?;
    let images = device.swapchain_khr().get_swapchain_images(swapchain)?;

    Ok((swapchain, images))
}

unsafe fn create_image_views(
    device: &ash::Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    let create_info = vk::ImageViewCreateInfo::default()
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .base_array_layer(0)
                .level_count(1)
                .layer_count(1),
        );

    let image_views = images
        .iter()
        .copied()
        .map(|image| {
            let create_info = create_info.image(image);
            Ok(device.create_image_view(&create_info, None)?)
        })
        .try_collect_vec()?;

    Ok(image_views)
}

unsafe fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    depth_buffer_view: vk::ImageView,
    swapchain_views: &[vk::ImageView],
) -> Result<Vec<vk::Framebuffer>> {
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .width(extent.width)
        .height(extent.height)
        .layers(1);

    let framebuffers = swapchain_views
        .iter()
        .copied()
        .map(|swapchain_view| {
            let attachments = &[swapchain_view, depth_buffer_view];
            let create_info = create_info.attachments(attachments);
            Ok(device.create_framebuffer(&create_info, None)?)
        })
        .try_collect_vec()?;

    Ok(framebuffers)
}
