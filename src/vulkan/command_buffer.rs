use crate::Result;

use ash::vk;
use std::ops::Deref;

pub struct CommandBufferOneTimeSubmit<'a> {
    inner: vk::CommandBuffer,
    device: &'a ash::Device,
    command_pool: &'a vk::CommandPool,
}

impl Deref for CommandBufferOneTimeSubmit<'_> {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for CommandBufferOneTimeSubmit<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_command_buffers(*self.command_pool, &[self.inner]);
        }
    }
}

impl<'a> CommandBufferOneTimeSubmit<'a> {
    pub unsafe fn new(
        device: &'a ash::Device,
        command_pool: &'a vk::CommandPool,
    ) -> Result<CommandBufferOneTimeSubmit<'a>> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(*command_pool)
            .command_buffer_count(1);

        let inner = device
            .allocate_command_buffers(&alloc_info)?
            .into_iter()
            .next()
            .expect("expected exactly one command buffer");

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(inner, &begin_info)?;

        Ok(CommandBufferOneTimeSubmit {
            inner,
            device,
            command_pool,
        })
    }
    pub unsafe fn submit(self, queue: vk::Queue, fence: vk::Fence) -> Result<()> {
        self.device.end_command_buffer(self.inner)?;

        let command_buffers = &[self.inner];
        let submits = &[vk::SubmitInfo::default().command_buffers(command_buffers)];

        self.device.queue_submit(queue, submits, fence)?;
        self.device.queue_wait_idle(queue)?;

        Ok(())
    }
}
