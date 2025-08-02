import open3d as o3d

# Load the mesh
mesh = o3d.io.read_triangle_mesh("output/person1.obj")
mesh.compute_vertex_normals()

# Open interactive viewer (you can rotate with mouse)
o3d.visualization.draw_geometries(
    [mesh],
    window_name="3D Face Viewer",
    width=800,
    height=600,
    mesh_show_back_face=True
)
