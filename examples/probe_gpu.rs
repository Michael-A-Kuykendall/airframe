// TEMPORARY GPU reachability + limit probe. Deleted after use.
use wgpu::Backends;

fn main() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("NO_ADAPTER: wgpu could not find a GPU");

        let info = adapter.get_info();
        println!("ADAPTER_NAME: {}", info.name);
        println!("ADAPTER_BACKEND: {:?}", info.backend);
        println!(
            "ADAPTER_MAX_STORAGE_BUFFER_BINDING_SIZE: {}",
            adapter.limits().max_storage_buffer_binding_size
        );
        println!(
            "ADAPTER_MAX_BUFFER_SIZE: {}",
            adapter.limits().max_buffer_size
        );

        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("NO_DEVICE: wgpu could not create a device");

        println!(
            "DEVICE_OK_MAX_STORAGE_BUFFER_BINDING_SIZE: {}",
            device.limits().max_storage_buffer_binding_size
        );
        println!("GPU_REACHABLE: true");
    });
}
