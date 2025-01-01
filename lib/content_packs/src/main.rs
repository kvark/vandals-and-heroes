extern crate core;

use crate::game::Game;
use std::path::Path;

mod definitions;
mod content_pack;
mod game;
mod templates;
mod instances;
mod camera_controller;

fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut game = Game::new(&event_loop, Path::new("data/packs/root"));
    game.load_level("main");

    #[allow(deprecated)] //TODO
    event_loop
        .run(|event, target| match event {
            winit::event::Event::NewEvents( winit::event::StartCause::ResumeTimeReached {..}) => {
                game.window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match game.on_event(&event) {
                Ok(control_flow) => {
                    target.set_control_flow(control_flow);
                }
                Err(_) => {
                    target.exit();
                }
            },
            _ => {}
        })
        .unwrap();
}
