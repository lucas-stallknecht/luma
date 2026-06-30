import datetime
import struct
from dataclasses import dataclass
from typing import cast

import bpy
import numpy as np
from mathutils import Vector
from numpy.typing import NDArray

HEADER_STRUCT = struct.Struct("16i")
MATERIAL_STRUCT = struct.Struct("3f 3i")
RENDERABLE_STRUCT = struct.Struct("16f 3i")
axis_fix = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


@dataclass
class MaterialData:
    base_color: Vector = Vector()
    base_color_tex_idx: int = -1
    metallic_roughness_tex_idx: int = -1
    normal_tex_idx: int = -1


@dataclass
class Renderable:
    transform: NDArray[np.float32]
    material_idx: int
    index_offset: int
    index_count: int


print("\n", str(datetime.datetime.now()))

positions: list[Vector] = []
normals: list[Vector] = []
tangents: list[Vector] = []
uvs: list[Vector] = []
indices: list[int] = []

textures: list[str] = []
texture_cache: dict[str, int] = {}  # texture name -> texture_idx

materials: list[MaterialData] = []
material_cache: dict[int, int] = {}  # material ptr -> material_idx

vertex_cache = {}  # (obj_idx, v_idx, loop_idx) -> new vertex index
next_vertex_index = 0

renderables: list[Renderable] = []  # final draw calls


def get_texture_idx(name: str | None) -> int:
    if not name:
        return -1

    if name in texture_cache:
        return texture_cache[name]

    idx = len(textures)
    textures.append(name)
    texture_cache[name] = idx
    return idx


def find_image_sources(
    socket: bpy.types.NodeSocket | None,
    visited: set[bpy.types.Node] | None = None,
) -> list[str]:
    if visited is None:
        visited = set()

    images: list[str] = []

    if not socket or not socket.is_linked:
        return images

    for link in socket.links:
        node: bpy.types.Node = link.from_node

        if node in visited:
            continue

        if isinstance(node, bpy.types.ShaderNodeTexImage) and node.image:
            images.append(node.image.name)
            continue

        # copy visited per-branch so sibling inputs don't block each other
        branch_visited = visited | {node}
        for inp in node.inputs:
            images.extend(find_image_sources(inp, branch_visited))

    return images


def extract_pbr_material(
    mat: bpy.types.Material,
) -> MaterialData | None:
    if not mat or not mat.node_tree:
        return None

    nodes = mat.node_tree.nodes
    result = MaterialData()

    for node in nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue

        bsdf: bpy.types.ShaderNodeBsdfPrincipled = node

        base_tex = find_image_sources(bsdf.inputs["Base Color"])
        if not base_tex:
            base_tex = find_image_sources(bsdf.inputs["Alpha"])
        result.base_color_tex_idx = get_texture_idx(base_tex[0]) if base_tex else -1
        result.base_color = Vector(bsdf.inputs["Base Color"].default_value[:3])

        # metallic and roughness may be shared
        mr_textures: set[str] = set()
        mr_textures.update(find_image_sources(bsdf.inputs["Metallic"]))
        mr_textures.update(find_image_sources(bsdf.inputs["Roughness"]))

        if mr_textures:
            # if multiple, just pick first (or later: ORM packing logic)
            result.metallic_roughness_tex_idx = get_texture_idx(next(iter(mr_textures)))

        normal_textures: list[str] = []
        for n in nodes:
            if isinstance(n, bpy.types.ShaderNodeNormalMap):
                normal_textures.extend(find_image_sources(n.inputs["Color"]))
        if normal_textures:
            result.normal_tex_idx = get_texture_idx(normal_textures[0])

        break

    return result


def get_material_idx(mat: bpy.types.Material | None) -> int:
    if mat is None or not mat.use_nodes or not mat.node_tree:
        return -1

    key = mat.as_pointer()
    if key in material_cache:
        return material_cache[key]

    new_material = extract_pbr_material(mat)
    if not new_material:
        return -1

    idx = len(materials)
    material_cache[key] = idx
    materials.append(new_material)

    return idx


for obj_idx, obj in enumerate(bpy.context.selected_objects):
    if obj.type != "MESH":
        continue

    print("Processing:", obj.name)

    mesh = cast(bpy.types.Mesh, obj.data)
    if not mesh.uv_layers.active:
        continue
    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers else None
    mesh.calc_tangents()

    # object transform
    # transform = np.array(obj.matrix_world, dtype=np.float32)
    transform = axis_fix @ np.array(obj.matrix_world, dtype=np.float32)

    # per-object grouping: (material_idx) -> indices
    submesh_map: dict[int, list[int]] = {}

    for tri in mesh.loop_triangles:
        mat = (
            obj.material_slots[tri.material_index].material
            if obj.material_slots
            else None
        )
        material_idx = get_material_idx(mat)

        if material_idx not in submesh_map:
            submesh_map[material_idx] = []

        tri_indices: list[int] = []

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
                tangents.append(Vector((*loop.tangent, loop.bitangent_sign)))

                if uv_layer:
                    uvs.append(uv_layer[loop_idx].uv.copy())
                else:
                    uvs.append(Vector((0.0, 0.0)))

            tri_indices.append(vertex_cache[key])

        submesh_map[material_idx].extend(tri_indices)

    # build renderables
    offset = len(indices)

    for material_idx, tri_indices in submesh_map.items():
        index_count = len(tri_indices)

        renderables.append(
            Renderable(
                transform=transform,
                material_idx=material_idx,
                index_offset=offset,
                index_count=index_count,
            )
        )

        indices.extend(tri_indices)
        offset += index_count

# final packed buffers
position_buffer = np.array(positions, dtype=np.float32)
normal_buffer = np.array(normals, dtype=np.float32)
tangent_buffer = np.array(tangents, dtype=np.float32)
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
# textures_offset, texture_size
# materials_offset, materials_size
# renderables_offset, renderables_size
#
# -- associated byte data
# positions   (vec3)
# normals     (vec3)
# tangents    (vec3)
# uvs         (vec2)
# indices     (u32)
# textures    (strings split by ,)
# materials   (vec3 + 3 * i32)
# renderables (mat4 + draw info)

texture_string = ",".join(f'"{name}"' for name in textures)
texture_bytes = texture_string.encode("utf-8")

# materials
material_bytes = bytearray()
for m in materials:
    material_bytes.extend(
        MATERIAL_STRUCT.pack(
            *m.base_color,
            m.base_color_tex_idx,
            m.metallic_roughness_tex_idx,
            m.normal_tex_idx,
        )
    )

# transform (mat4 = 16 * f32)
# material_idx (i32)
# index_offset (i32)
# index_count (i32)
renderable_bytes = bytearray()
for r in renderables:
    renderable_bytes.extend(
        RENDERABLE_STRUCT.pack(
            *r.transform.flatten(),
            r.material_idx,
            r.index_offset,
            r.index_count,
        )
    )

pos_bytes = position_buffer.tobytes()
normal_bytes = normal_buffer.tobytes()
tangent_bytes = tangent_buffer.tobytes()
uv_bytes = uv_buffer.tobytes()
index_bytes = index_buffer.tobytes()

# write file
filename = bpy.path.abspath("//scene.bin")

with open(filename, "wb") as f:
    _ = f.write(b"\x00" * HEADER_STRUCT.size)

    positions_offset = f.tell()
    _ = f.write(pos_bytes)
    positions_size = len(pos_bytes)

    normals_offset = f.tell()
    _ = f.write(normal_bytes)
    normals_size = len(normal_bytes)

    tangents_offset = f.tell()
    _ = f.write(tangent_bytes)
    tangents_size = len(tangent_bytes)

    uvs_offset = f.tell()
    _ = f.write(uv_bytes)
    uvs_size = len(uv_bytes)

    indices_offset = f.tell()
    _ = f.write(index_bytes)
    indices_size = len(index_bytes)

    textures_offset = f.tell()
    _ = f.write(texture_bytes)
    textures_size = len(texture_bytes)

    materials_offset = f.tell()
    _ = f.write(material_bytes)
    materials_size = len(material_bytes)

    renderables_offset = f.tell()
    _ = f.write(renderable_bytes)
    renderables_size = len(renderable_bytes)

    _ = f.seek(0)
    _ = f.write(
        HEADER_STRUCT.pack(
            positions_offset,
            positions_size,
            normals_offset,
            normals_size,
            tangents_offset,
            tangents_size,
            uvs_offset,
            uvs_size,
            indices_offset,
            indices_size,
            textures_offset,
            textures_size,
            materials_offset,
            materials_size,
            renderables_offset,
            renderables_size,
        )
    )

print("- file export done")
