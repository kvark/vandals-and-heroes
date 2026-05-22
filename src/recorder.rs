use crate::config;
use crate::physics::Physics;
use rapier3d::dynamics::RigidBodyHandle;
use std::{fs, io::BufWriter, path::Path};

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct ObjectSnapshot {
    pub name: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub linvel: [f32; 3],
    pub angvel: [f32; 3],
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Snapshot {
    pub tick: u64,
    pub time: f32,
    pub objects: Vec<ObjectSnapshot>,
}

pub struct Recorder {
    writer: BufWriter<fs::File>,
    format: config::RecorderFormat,
    tick: u64,
}

impl Recorder {
    pub fn new(cfg: &config::Recorder) -> Self {
        let path: &Path = cfg.path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let file = fs::File::create(path)
            .unwrap_or_else(|e| panic!("Unable to open recorder file {path:?}: {e}"));
        log::info!("Recording state to {path:?} as {:?}", cfg.format);
        Self {
            writer: BufWriter::new(file),
            format: cfg.format,
            tick: 0,
        }
    }

    pub fn record<'a, I>(&mut self, time: f32, physics: &Physics, bodies: I)
    where
        I: IntoIterator<Item = (&'a str, RigidBodyHandle)>,
    {
        let objects = bodies
            .into_iter()
            .filter_map(|(name, handle)| {
                physics.body_kinematics(handle).map(|k| ObjectSnapshot {
                    name: name.to_owned(),
                    translation: k.translation,
                    rotation: k.rotation,
                    linvel: k.linvel,
                    angvel: k.angvel,
                })
            })
            .collect();
        let snapshot = Snapshot {
            tick: self.tick,
            time,
            objects,
        };
        self.write(&snapshot);
        self.tick += 1;
    }

    fn write(&mut self, snapshot: &Snapshot) {
        use std::io::Write;
        match self.format {
            config::RecorderFormat::Ron => {
                let line = ron::ser::to_string(snapshot).expect("ron serialize");
                writeln!(self.writer, "{}", line).expect("recorder write");
            }
            config::RecorderFormat::Bincode => {
                bincode::serde::encode_into_std_write(
                    snapshot,
                    &mut self.writer,
                    bincode::config::standard(),
                )
                .expect("bincode write");
            }
        }
        // Flush every tick so the log survives a crash or SIGKILL —
        // the whole point is to debug physics issues that may panic.
        self.writer.flush().expect("recorder flush");
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        use std::io::Write;
        let _ = self.writer.flush();
    }
}
