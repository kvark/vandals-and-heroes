use std::ops::Range;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransformDesc {
    #[serde(default = "TransformDesc::default_position")]
    pub position: (f32, f32, f32),
    #[serde(default = "TransformDesc::default_rotation")]
    pub rotation: (f32, f32, f32),
    #[serde(default = "TransformDesc::default_scale")]
    pub scale: (f32, f32, f32),
}

impl TransformDesc {
    pub fn default_position() -> (f32, f32, f32) {
        (0.0, 0.0, 0.0)
    }
    
    pub fn default_rotation() -> (f32, f32, f32) {
        (0.0, 0.0, 0.0)
    }
    
    pub fn default_scale() -> (f32, f32, f32) {
        (1f32, 1f32, 1f32)
    }
}

impl Default for TransformDesc {
    fn default() -> TransformDesc {
        Self {
            position: TransformDesc::default_position(),
            rotation: TransformDesc::default_rotation(),
            scale: TransformDesc::default_scale(),
        }
    }
}

impl From<TransformDesc> for nalgebra::Isometry3<f32>{
    fn from(value: TransformDesc) -> Self {
        let (x, y, z) = value.position;
        let (roll, pitch, yaw) = value.rotation;
        let position = nalgebra::Point3::new(x, y, z);
        let rotation = nalgebra::UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        nalgebra::Isometry3::from_parts(position.into(), rotation)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum ShapeDesc {
    Box {
        size: (f32, f32, f32)
    },
    Sphere {
        radius: f32
    },
    Mesh {
        path: PathBuf,
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ColliderDesc {
    pub shape: ShapeDesc,
    pub transform: TransformDesc,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum PhysicsBodyDesc {
    RigidBody {
        mass: f32,
    },
    StaticBody
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PhysicsDesc {
    pub body: PhysicsBodyDesc,
    pub colliders: Vec<ColliderDesc>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ObjectDesc {
    pub id: String,
    pub scene_path: Option<PathBuf>,
    pub physics: Option<PhysicsDesc>,
    pub script_path: Option<PathBuf>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HeightMapDesc {
    pub id: String,
    pub image_path: PathBuf,
    pub radius: Range<f32>,
    pub density: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LevelObjectDesc {
    pub id: String,
    pub entity_id: String,
    #[serde(default)]
    pub transform: TransformDesc
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LevelDesc {
    pub id: String,
    pub name: String,
    pub height_map: HeightMapDesc,
    pub objects: Vec<LevelObjectDesc>
}











