import datetime
import struct

import bpy
import numpy as np

HEADER_STRUCT = struct.Struct("12I")
MATERIAL_STRUCT = struct.Struct("4f")
RENDERABLE_STRUCT = struct.Struct("16f3I")

print("\n", str(datetime.datetime.now()))

positions = []
normals = []
uvs = []
indices = []

materials = []
material_cache = {}  # material ptr -> material_idx

vertex_cache = {}  # (obj_idx, v_idx, loop_idx) -> new vertex index
next_vertex_index = 0

renderables = []  # final draw calls


def get_material_idx(mat):
    if mat is None:
        return -1

    key = mat.as_pointer()
    if key in material_cache:
        return material_cache[key]

    idx = len(materials)
    material_cache[key] = idx

    color = mat.diffuse_color[:]
    materials.append({"color": (color[0], color[1], color[2], color[3])})

    return idx


for obj_idx, obj in enumerate(bpy.context.selected_objects):
    if obj.type != "MESH":
        continue

    print("Processing:", obj.name)

    mesh = obj.data
    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers else None

    # object transform
    transform = np.array(obj.matrix_world, dtype=np.float32)

    # per-object grouping: (material_idx) -> indices
    submesh_map = {}

    for tri in mesh.loop_triangles:
        mat = (
            obj.material_slots[tri.material_index].material
            if obj.material_slots
            else None
        )
        material_idx = get_material_idx(mat)

        if material_idx not in submesh_map:
            submesh_map[material_idx] = []

        tri_indices = []

        for loop_idx in tri.loops:
            loop = mesh.loops[loop_idx]
            v_idx = loop.vertex_index

            key = (obj_idx, v_idx, loop_idx)

            if key not in vertex_cache:
                vertex_cache[key] = next_vertex_index
                next_vertex_index += 1

                positions.append(mesh.vertices[v_idx].co.copy())

                normals.append(
                    loop.normal.copy()
                    if mesh.has_custom_normals
                    else mesh.vertices[v_idx].normal
                )

                if uv_layer:
                    uvs.append(uv_layer[loop_idx].uv.copy())
                else:
                    uvs.append((0.0, 0.0))

            tri_indices.append(vertex_cache[key])

        submesh_map[material_idx].extend(tri_indices)

    # build renderables
    offset = len(indices)

    for material_idx, tri_indices in submesh_map.items():
        index_count = len(tri_indices)

        renderables.append(
            {
                "transform": transform,
                "material_idx": material_idx,
                "index_offset": offset,
                "index_count": index_count,
            }
        )

        indices.extend(tri_indices)
        offset += index_count

# final packed buffers
position_buffer = np.array(positions, dtype=np.float32)
normal_buffer = np.array(normals, dtype=np.float32)
uv_buffer = np.array(uvs, dtype=np.float32)
index_buffer = np.array(indices, dtype=np.uint32)

print("- building done")
print("Vertices:", len(position_buffer))
print("Indices:", len(index_buffer))
print("Materials:", len(materials))
print("Draw calls:", len(renderables))

# file format
#
# -- header layout (all u32)
# positions_offset, positions_size
# normals_offset, normals_size
# uvs_offset, uvs_size
# indices_offset, indices_size
# materials_offset, materials_size
# renderables_offset, renderables_size
#
# -- associated byte data
# positions   (vec3)
# normals     (vec3)
# uvs         (vec2)
# indices     (u32)
# materials   (vec4)
# renderables (mat4 + draw info)

# materials (vec4)
material_bytes = bytearray()
for m in materials:
    material_bytes.extend(MATERIAL_STRUCT.pack(*m["color"]))

# transform (mat4 = 16 * f32)
# material_idx (u32)
# index_offset (u32)
# index_count (u32)
renderable_bytes = bytearray()
for r in renderables:
    renderable_bytes.extend(
        RENDERABLE_STRUCT.pack(
            *r["transform"].flatten(),
            r["material_idx"],
            r["index_offset"],
            r["index_count"],
        )
    )

pos_bytes = position_buffer.tobytes()
normal_bytes = normal_buffer.tobytes()
uv_bytes = uv_buffer.tobytes()
index_bytes = index_buffer.tobytes()

# write file
filename = bpy.path.abspath("//scene.bin")

with open(filename, "wb") as f:
    f.write(b"\x00" * HEADER_STRUCT.size)

    positions_offset = f.tell()
    f.write(pos_bytes)
    positions_size = len(pos_bytes)

    normals_offset = f.tell()
    f.write(normal_bytes)
    normals_size = len(normal_bytes)

    uvs_offset = f.tell()
    f.write(uv_bytes)
    uvs_size = len(uv_bytes)

    indices_offset = f.tell()
    f.write(index_bytes)
    indices_size = len(index_bytes)

    materials_offset = f.tell()
    f.write(material_bytes)
    materials_size = len(material_bytes)

    renderables_offset = f.tell()
    f.write(renderable_bytes)
    renderables_size = len(renderable_bytes)

    f.seek(0)

    f.write(
        HEADER_STRUCT.pack(
            positions_offset,
            positions_size,
            normals_offset,
            normals_size,
            uvs_offset,
            uvs_size,
            indices_offset,
            indices_size,
            materials_offset,
            materials_size,
            renderables_offset,
            renderables_size,
        )
    )

print("- file export done")
