mod config;

use blade_graphics as gpu;
use std::{fs, time};

struct Game {
    // engine stuff
    choir: choir::Choir,
    command_encoder: gpu::CommandEncoder,
    last_sync_point: Option<gpu::SyncPoint>,
    gpu_surface: gpu::Surface,
    gpu_context: gpu::Context,
    // windowing
    window: winit::window::Window,
    // game data
}

struct QuitEvent;

impl Game {
    fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        log::info!("Initializing");

        let config: config::Main = ron::de::from_bytes(
            &fs::read("data/config.ron").expect("Unable to open the main config"),
        )
        .expect("Unable to parse the main config");

        let choir = choir::Choir::default();
        let gpu_context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                ..Default::default()
            })
        }
        .expect("Unable to initialize GPU");
        let command_encoder = gpu_context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        {
            log::info!("Loading map: {}", config.map);
            let png_path = format!("data/maps/{}/map.png", config.map);
            let decoder = png::Decoder::new(fs::File::open(png_path).unwrap());
            let mut reader = decoder.read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).unwrap();
            let _bytes = &buf[..info.buffer_size()];
        }

        let window_attributes =
            winit::window::Window::default_attributes().with_title("Vandals and Heroes");
        let window = event_loop.create_window(window_attributes).unwrap();
        let window_size = window.inner_size();

        let surface_config = gpu::SurfaceConfig {
            size: gpu::Extent {
                width: window_size.width,
                height: window_size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Recent,
            ..Default::default()
        };
        let gpu_surface = gpu_context
            .create_surface_configured(&window, surface_config)
            .unwrap();

        Self {
            choir,
            command_encoder,
            last_sync_point: None,
            gpu_surface,
            gpu_context,
            window,
        }
    }

    fn wait_for_gpu(&mut self) {
        if let Some(sync_point) = self.last_sync_point.take() {
            self.gpu_context.wait_for(&sync_point, 0);
        }
    }

    fn redraw(&mut self) -> time::Duration {
        let frame = self.gpu_surface.acquire_frame();
        self.command_encoder.start();
        if let _pass = self.command_encoder.render(
            "draw",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame.texture_view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            //TODO
        }
        self.command_encoder.present(frame);
        let sync_point = self.gpu_context.submit(&mut self.command_encoder);
        self.wait_for_gpu();
        self.last_sync_point = Some(sync_point);
        time::Duration::from_millis(16)
    }

    fn on_event(
        &mut self,
        event: &winit::event::WindowEvent,
    ) -> Result<winit::event_loop::ControlFlow, QuitEvent> {
        match *event {
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => match key_code {
                winit::keyboard::KeyCode::Escape => {
                    return Err(QuitEvent);
                }
                _ => {}
            },
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                        state: winit::event::ElementState::Released,
                        ..
                    },
                ..
            } => match key_code {
                _ => {}
            },
            winit::event::WindowEvent::CloseRequested => {
                return Err(QuitEvent);
            }
            winit::event::WindowEvent::RedrawRequested => {
                let wait = self.redraw();

                return Ok(
                    if let Some(repaint_after_instant) = std::time::Instant::now().checked_add(wait)
                    {
                        winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
                    } else {
                        winit::event_loop::ControlFlow::Wait
                    },
                );
            }
            _ => {}
        }

        Ok(winit::event_loop::ControlFlow::Poll)
    }
}

impl Drop for Game {
    fn drop(&mut self) {
        self.wait_for_gpu();
    }
}

fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut game = Game::new(&event_loop);

    event_loop
        .run(|event, target| match event {
            winit::event::Event::AboutToWait => {
                game.window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match game.on_event(&event) {
                Ok(control_flow) => {
                    target.set_control_flow(control_flow);
                }
                Err(QuitEvent) => {
                    target.exit();
                }
            },
            _ => {}
        })
        .unwrap();
}
