use crate::util::IteratorExt;
use crate::Result;

use ash::vk;

#[derive(Debug)]
pub struct SyncObjectStore {
    in_flight: Vec<vk::Fence>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
}

impl SyncObjectStore {
    pub unsafe fn new(device: &ash::Device, frame_count: usize) -> Result<Self> {
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let in_flight = std::iter::repeat(fence_info)
            .take(frame_count)
            .map(|create_info| Ok(device.create_fence(&create_info, None)?))
            .try_collect_vec()?;

        let image_available = std::iter::repeat(vk::SemaphoreCreateInfo::default())
            .take(frame_count)
            .map(|create_info| Ok(device.create_semaphore(&create_info, None)?))
            .try_collect_vec()?;

        let render_finished = std::iter::repeat(vk::SemaphoreCreateInfo::default())
            .take(frame_count)
            .map(|create_info| Ok(device.create_semaphore(&create_info, None)?))
            .try_collect_vec()?;

        Ok(Self {
            in_flight,
            image_available,
            render_finished,
        })
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for fence in std::mem::take(&mut self.in_flight) {
            device.destroy_fence(fence, None);
        }

        let image_available = std::mem::take(&mut self.image_available);
        let render_finished = std::mem::take(&mut self.render_finished);
        for semaphore in [image_available, render_finished].into_iter().flatten() {
            device.destroy_semaphore(semaphore, None);
        }
    }

    pub fn get_in_flight(&self, frame_idx: u32) -> vk::Fence {
        self.in_flight[frame_idx as usize]
    }

    pub fn get_image_available(&self, frame_idx: u32) -> vk::Semaphore {
        self.image_available[frame_idx as usize]
    }

    pub fn get_render_finished(&self, frame_idx: u32) -> vk::Semaphore {
        self.render_finished[frame_idx as usize]
    }
}
