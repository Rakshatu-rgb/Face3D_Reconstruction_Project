import numpy as np

def reconstruct_mesh(flame_model, shape_coeffs=None, expr_coeffs=None):
    # Extract from FLAME dictionary
    mean_shape = flame_model['v_template']             # [N, 3]
    shape_basis = flame_model['shapedirs']             # [N, 3, num_shape]
    expr_basis = flame_model['posedirs']               # [N, 3, num_expr]
    faces = flame_model.get('faces') or flame_model.get('f')
    # [F, 3]
    print("Shape basis shape:", shape_basis.shape)
    print("Shape coeffs shape:", shape_coeffs.shape)

    if shape_coeffs is None:
        shape_coeffs = np.random.randn(shape_basis.shape[2]) * 0.03
    if expr_coeffs is None:
        expr_coeffs = np.random.randn(expr_basis.shape[2]) * 0.03
    print("Expression basis shape:", expr_basis.shape)
    print("Expression coeffs shape:", expr_coeffs.shape)

    # Safely crop shape basis to match shape_coeffs length
    num_shape = shape_coeffs.shape[0]
    shape_offset = np.tensordot(shape_basis[:, :, :num_shape], shape_coeffs, axes=([2], [0]))
    expr_offset = np.tensordot(expr_basis, expr_coeffs, axes=([2], [0]))     # [N, 3]

    vertices = mean_shape + shape_offset + expr_offset
    return vertices, faces
