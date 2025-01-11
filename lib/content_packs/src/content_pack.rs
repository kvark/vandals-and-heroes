use crate::definitions::{LevelDesc, ObjectDesc};
use ron::de::from_reader;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct ContentPack {
    entities: HashMap<String, ObjectDesc>,
    levels: Vec<LevelDesc>,
    pub(crate) directory: PathBuf,
}

impl ContentPack {
    pub fn new(directory: &Path) -> Self {
        let entities_vec: Vec<ObjectDesc> = Self::read_ron(&directory.join("entities.ron"));
        let entities: HashMap<String, ObjectDesc> = entities_vec
            .into_iter()
            .map(|e| (e.id.clone(), e))
            .collect();

        let mut levels = Vec::new();
        let levels_directory = directory.join("levels");
        for entry in std::fs::read_dir(levels_directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "ron" {
                    levels.push(Self::read_ron(&path));
                }
            }
        }

        Self {
            entities,
            levels,
            directory: directory.to_path_buf(),
        }
    }

    fn read_ron<T: DeserializeOwned>(path: &Path) -> T {
        let file = File::open(path).unwrap();
        from_reader(file).unwrap()
    }

    pub fn get_entity_by_id(&self, id: &str) -> Option<&ObjectDesc> {
        self.entities.get(id)
    }

    pub fn get_resource_path(&self, path: &Path) -> PathBuf {
        self.directory.join(path)
    }

    pub fn get_level_by_id(&self, level_id: &str) -> Option<&LevelDesc> {
        self.levels.iter().find(|l| l.id == level_id)
    }
}
