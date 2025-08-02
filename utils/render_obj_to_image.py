import os
import trimesh
import pyrender
import numpy as np
import imageio
from tqdm import tqdm

def render_rotated(obj_path, output_path="renders/rotation.gif", num_frames=60):
    print(f"📦 Rendering rotating view of: {os.path.basename(obj_path)}")

    # Load mesh
    mesh = trimesh.load(obj_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    # Fix scale if too small
    mesh.apply_scale(1.5)

    # Create pyrender scene
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2, roughnessFactor=0.8, baseColorFactor=[0.5, 0.5, 0.5, 1.0]
    )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh)

    # Add rotating camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_node = scene.add(camera, pose=np.eye(4))

    # Add lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=np.eye(4))

    # Offscreen renderer
    r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)

    # Generate frames
    images = []
    for angle in tqdm(range(num_frames), desc="Rendering"):
        theta = 2 * np.pi * angle / num_frames
        cam_pose = np.array([
            [np.cos(theta), 0, np.sin(theta), 0.3 * np.sin(theta)],
            [0, 1, 0, 0.1],
            [-np.sin(theta), 0, np.cos(theta), 0.3 * np.cos(theta)],
            [0, 0, 0, 1]
        ])
        scene.set_pose(cam_node, pose=cam_pose)
        color, _ = r.render(scene)
        images.append(color)

    r.delete()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, images, duration=0.05)
    print(f"✅ Saved rotating view as {output_path}")

# Run the rendering for one of your meshes (e.g., person1.obj)
if __name__ == "__main__":
    render_rotated("output/person1.obj")
