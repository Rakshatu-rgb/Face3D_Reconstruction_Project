def save_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces + 1:  # .obj is 1-indexed
            f.write(f'f {face[0]} {face[1]} {face[2]}\n')
