//! Original low-poly landmarks for the two Heroes supply stops.

use blade_graphics as gpu;
use nalgebra::{Matrix4, Point3, Vector3};
use vandals_and_heroes::{GeometryDesc, MaterialDesc, ModelDesc, VertexDesc};

pub enum Kind {
    Depot,
    Ridge,
}

fn face(mesh: &mut GeometryDesc, mut corners: [Point3<f32>; 3], center: Point3<f32>) {
    let mut normal = (corners[1] - corners[0]).cross(&(corners[2] - corners[0]));
    if normal.dot(&(corners[0] - center)) < 0.0 {
        corners.swap(1, 2);
        normal = -normal;
    }
    let first = mesh.vertices.len() as u32;
    for pos in corners {
        mesh.vertices.push(VertexDesc {
            pos,
            normal: normal.normalize(),
            ..Default::default()
        });
    }
    mesh.indices.push([first, first + 1, first + 2]);
}

fn block(mesh: &mut GeometryDesc, [x, y, z]: [f32; 3], [a, b, c]: [f32; 3]) {
    let center = Point3::new(x, y, z);
    let points = [
        [-a, -b, -c],
        [a, -b, -c],
        [a, b, -c],
        [-a, b, -c],
        [-a, -b, c],
        [a, -b, c],
        [a, b, c],
        [-a, b, c],
    ]
    .map(|[dx, dy, dz]| center + Vector3::new(dx, dy, dz));
    for [i, j, k, l] in [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 2, 6, 7],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ] {
        face(mesh, [points[i], points[j], points[k]], center);
        face(mesh, [points[i], points[k], points[l]], center);
    }
}

pub fn model(kind: Kind) -> ModelDesc {
    let geometry = |name: &str, material_index| GeometryDesc {
        name: name.to_owned(),
        vertices: Vec::new(),
        indices: Vec::new(),
        index_type: Some(gpu::IndexType::U32),
        transform: Matrix4::identity(),
        material_index,
    };
    let mut frame = geometry("salvaged frame", 1);
    let mut emblem = geometry("signal enamel", 2);
    let accent = match kind {
        Kind::Depot => {
            // Low, broad crate with two contrasting reinforced straps.
            block(&mut frame, [0.0, -0.9, 0.0], [1.15, 0.62, 0.65]);
            for x in [-0.8, 0.8] {
                block(&mut emblem, [x, -0.86, 0.0], [0.10, 0.68, 0.68]);
            }
            // Short box-shaped radio cap.
            block(&mut emblem, [0.0, 0.0, 0.0], [0.35, 0.23, 0.35]);
            [0.13, 0.86, 0.81, 1.0]
        }
        Kind::Ridge => {
            // Narrow, taller cairn with a diamond crown rather than a crate.
            block(&mut frame, [0.0, -1.28, 0.0], [0.78, 0.24, 0.78]);
            block(&mut frame, [0.0, -0.49, 0.0], [0.16, 0.72, 0.16]);
            let center = Point3::new(0.0, 0.55, 0.0);
            let ring = [
                Point3::new(0.47, 0.55, 0.0),
                Point3::new(0.0, 0.55, 0.47),
                Point3::new(-0.47, 0.55, 0.0),
                Point3::new(0.0, 0.55, -0.47),
            ];
            for i in 0..ring.len() {
                let edge = [ring[i], ring[(i + 1) % ring.len()]];
                for tip in [-0.44, 1.64] {
                    face(
                        &mut emblem,
                        [edge[0], edge[1], Point3::new(0.0, tip, 0.0)],
                        center,
                    );
                }
            }
            [0.77, 0.29, 0.94, 1.0]
        }
    };
    let material = |name: &str, color| MaterialDesc {
        name: Some(name.to_owned()),
        base_color_factor: color,
        ..Default::default()
    };
    ModelDesc {
        geometries: vec![frame, emblem],
        materials: vec![
            MaterialDesc::default(),
            material("oxidized alloy", [0.24, 0.21, 0.17, 1.0]),
            material("signal enamel", accent),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_have_distinct_profiles_and_valid_faces() {
        let depot = model(Kind::Depot);
        let ridge = model(Kind::Ridge);
        let tallest = |model: &ModelDesc| {
            model
                .positions()
                .into_iter()
                .map(|p| p.y)
                .fold(f32::NEG_INFINITY, f32::max)
        };
        assert!(tallest(&ridge) > tallest(&depot) + 1.0);
        for model in [depot, ridge] {
            for geom in model.geometries {
                assert!(!geom.indices.is_empty());
                for triangle in geom.indices {
                    let [a, b, c] = triangle.map(|i| geom.vertices[i as usize]);
                    let geometric = (b.pos - a.pos).cross(&(c.pos - a.pos));
                    assert!(geometric.dot(&a.normal) > 0.0);
                    assert!((a.normal.norm() - 1.0).abs() < 1e-5);
                }
            }
        }
    }
}
