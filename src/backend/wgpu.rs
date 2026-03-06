use wgpu::{Device, Queue};

pub struct WgpuContext {
    pub device: Device,
    pub queue: Queue,
}

impl WgpuContext {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        println!("Selected Adapter: {:?}", adapter.get_info());
        let limits = adapter.limits();
        println!(
            "Adapter Limits: max_buffer_size={}, max_storage_buffer_binding_size={}",
            limits.max_buffer_size, limits.max_storage_buffer_binding_size
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        Self { device, queue }
    }
}
