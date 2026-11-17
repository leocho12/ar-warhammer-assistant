"""Read OBJ and Unity bundle geometry without loading executable mod content."""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from trimesh.exchange.obj import load_obj
import UnityPy
from UnityPy.environment import Environment
from UnityPy.helpers.CompressionHelper import DECOMPRESSION_MAP
from UnityPy.helpers.MeshHelper import MeshHandler
from UnityPy.helpers.UnityVersion import UnityVersion
from UnityPy.streams import EndianBinaryReader


@dataclass
class Geometry:
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray


@dataclass
class Surface:
    geometry: str
    transform: np.ndarray
    texture: str | None = None
    color: tuple = (1.0, 1.0, 1.0, 1.0)
    uv_transform: tuple = (1.0, 1.0, 0.0, 0.0)
    cutout: float = 0.5


def identity():
    return np.eye(4, dtype=np.float32)


def translation(x, y, z):
    result = identity()
    result[:3, 3] = x, y, z
    return result


def scale(x, y, z):
    return np.diag([x, y, z, 1]).astype(np.float32)


def euler(x, y, z):
    x, y, z = np.deg2rad([x, y, z])
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    result = identity()
    result[:3, :3] = ry @ rx @ rz
    return result


def quaternion(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    result = identity()
    result[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    return result


def tts_transform(values, root):
    r = [float(values.get(f"rot{axis}", 0.0)) for axis in "XYZ"]
    s = [float(values.get(f"scale{axis}", 1.0)) for axis in "XYZ"]
    p = [float(values.get(f"pos{axis}", 0.0)) for axis in "XYZ"]
    if root:
        return euler(r[0], 0.0, r[2]) @ scale(*s)
    return translation(*p) @ euler(*r) @ scale(*s)


def geometry(vertices, faces, normals, uv):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if not len(vertices) or not len(faces) or not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh has no finite triangles")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh has invalid indices")
    if normals is None:
        corners = vertices[faces]
        triangle_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals = np.zeros_like(vertices)
        for corner in faces.T:
            np.add.at(normals, corner, triangle_normals)
    normals = np.asarray(normals, dtype=np.float32)[:, :3]
    normals = np.nan_to_num(normals)
    normals /= np.maximum(1e-16, np.linalg.norm(normals, axis=1, keepdims=True))
    if uv is None:
        uv = np.zeros((len(vertices), 2), dtype=np.float32)
    uv = np.nan_to_num(np.asarray(uv, dtype=np.float32)[:, :2])
    selected = vertices[np.unique(faces)]
    bounds = np.stack((selected.min(axis=0), selected.max(axis=0)))
    return Geometry(np.ascontiguousarray(np.column_stack([vertices, normals, uv])), faces, bounds)


def obj_geometry(path):
    with path.open("rb") as handle:
        parsed = load_obj(handle, skip_materials=True, group_material=False)
    for piece in parsed.get("geometry", {"mesh": parsed}).values():
        if not len(piece.get("faces", [])):
            continue
        visual = piece.get("visual")
        # Unity and TTS use a left-handed coordinate system. OBJ is right-handed.
        vertices = piece["vertices"].copy()
        vertices[:, 0] *= -1
        normals = piece.get("vertex_normals")
        if normals is not None:
            normals = normals.copy()
            normals[:, 0] *= -1
        faces = piece["faces"]
        if faces.shape[1] != 3:
            faces = np.concatenate([faces[:, [0, k, k + 1]] for k in range(1, faces.shape[1] - 1)])
        yield geometry(vertices, faces[:, ::-1], normals, getattr(visual, "uv", None))


def xyz(vector):
    return vector.x, vector.y, vector.z


def opened_image(path):
    with Image.open(path) as image:
        return image.convert("RGBA")


def unity_material(pointer):
    if pointer is None or not pointer.m_PathID:
        return None, (1., 1., 1., 1.), (1., 1., 0., 0.), 0.
    material = pointer.deref_parse_as_object()
    properties = material.m_SavedProperties
    colors = dict(properties.m_Colors)
    value = colors.get("_BaseColor", colors.get("_Color"))
    tint = (value.r, value.g, value.b, value.a) if value else (1., 1., 1., 1.)
    textures = dict(properties.m_TexEnvs)
    floats = dict(properties.m_Floats)
    cutoff = floats.get("_Cutoff", 0.5) if floats.get("_Mode") == 1 or floats.get("_AlphaClip") else 0.
    for name in ("_BaseMap", "_MainTex", "_BaseColorMap", "_Albedo", "_Diffuse"):
        slot = textures.get(name)
        if slot and slot.m_Texture.m_PathID:
            uv = (slot.m_Scale.x, slot.m_Scale.y, slot.m_Offset.x, slot.m_Offset.y)
            return slot.m_Texture, tint, uv, cutoff
    return None, tint, (1., 1., 0., 0.), cutoff


def bundle_members(path):
    """Read the UnityFS directory without inflating any mesh or texture data."""
    with path.open("rb") as stream:
        reader = EndianBinaryReader(stream)
        if reader.read_string_to_null() != "UnityFS":
            return []
        header_version = reader.read_u_int()
        reader.read_string_to_null()
        engine = UnityVersion.from_str(reader.read_string_to_null())
        reader.read_long()
        compressed, raw, flags = reader.read_u_int(), reader.read_u_int(), reader.read_u_int()
        if header_version >= 7 or engine[0] == 2019 and engine >= (2019, 4, 15):
            reader.align_stream(16)
        if flags & 0x80:
            reader.Position = reader.Length - compressed
        info = DECOMPRESSION_MAP[flags & 0x3f](reader.read_bytes(compressed), raw)
    blocks = EndianBinaryReader(info)
    blocks.Position = 16
    count = blocks.read_u_int()
    blocks.Position += 10 * count
    count = blocks.read_u_int()
    names = []
    for _ in range(count):
        blocks.Position += 20
        names.append(blocks.read_string_to_null().lower())
    return names


def catalog_bundles(root):
    members = {}
    for path in sorted((root / "_assets").glob("*/*.unity3d")):
        try:
            entries = bundle_members(path)
        except (OSError, ValueError, IndexError, KeyError, TypeError, struct.error):
            continue
        for name in entries:
            if name.startswith("cab-") and "." not in name:
                members.setdefault(name, str(path))
    return members


class BundleEnvironment(Environment):
    def __init__(self, path, library):
        super().__init__(str(path))
        self.library = library

    def find_file(self, name, is_dependency=True):
        if not self.get_cab(name) and name in self.library:
            self.load_file(self.library[name], is_dependency=True)
        return super().find_file(name, is_dependency)


def mesh_skin(renderer, mesh, handler, world, vertices, normals, placement):
    bones = getattr(renderer, "m_Bones", None)
    poses = mesh.m_BindPose
    if (not bones or not poses or len(bones) != len(poses)
            or handler.m_BoneIndices is None or handler.m_BoneWeights is None):
        return vertices, normals, placement
    matrices = []
    for pointer, pose in zip(bones, poses):
        bind_pose = np.array([[getattr(pose, f"e{i}{j}") for j in range(4)] for i in range(4)], dtype=np.float32)
        matrices.append(world(pointer.m_PathID) @ bind_pose if pointer.m_PathID else placement)
    matrices = np.asarray(matrices)
    indices = np.asarray(handler.m_BoneIndices).clip(0, len(matrices) - 1)
    weights = np.asarray(handler.m_BoneWeights, dtype=np.float32)
    total = weights.sum(axis=1)
    if len(weights) != len(vertices) or not np.allclose(total, 1, atol=0.04):
        return vertices, normals, placement
    transforms = (matrices[indices] * weights[:, :, None, None]).sum(axis=1)
    points = np.einsum("nij,nj->ni", transforms[:, :3, :3], vertices) + transforms[:, :3, 3]
    directions = np.einsum("nij,nj->ni", transforms[:, :3, :3], normals[:, :3]) if normals is not None else None
    return points, directions, identity()


def unity_bundle(path, dependencies, cache, prefix):
    env = BundleEnvironment(path, cache.bundle_library) if cache.bundle_library else UnityPy.load(str(path))
    primary = env.objects
    game_object_ids = {reader.path_id for reader in primary if reader.type.name == "GameObject"}
    prefab_ids = set()
    for reader in primary:
        if reader.type.name == "AssetBundle":
            for _name, info in reader.parse_as_object().m_Container:
                pointer = info.asset
                if not pointer.m_FileID and pointer.m_PathID in game_object_ids:
                    prefab_ids.add(pointer.m_PathID)
    for dependency in dependencies:
        env.load_file(str(dependency))
    objects = {}
    for reader in primary:
        if reader.type.name in {"Transform", "RectTransform", "GameObject", "MeshRenderer",
                                "MeshFilter", "SkinnedMeshRenderer"}:
            key = reader.type.name
            if key == "RectTransform":
                key = "Transform"
            objects.setdefault(key, {})[reader.path_id] = reader.parse_as_object()
    transforms = objects.get("Transform", {})
    transforms_by_object = {t.m_GameObject.m_PathID: k for k, t in transforms.items()}
    transform_roots = {transforms_by_object[k] for k in prefab_ids if k in transforms_by_object}
    filters = {obj.m_GameObject.m_PathID: obj for obj in objects.get("MeshFilter", {}).values()}
    game_objects = objects.get("GameObject", {})
    world_cache = {0: identity()}

    def world(key):
        if key not in world_cache:
            t = transforms[key]
            local = quaternion(t.m_LocalRotation) @ scale(*xyz(t.m_LocalScale))
            # TTS places the instantiated prefab root at the saved object's position.
            if t.m_Father.m_PathID:
                local = world(t.m_Father.m_PathID) @ translation(*xyz(t.m_LocalPosition)) @ local
            world_cache[key] = local
        return world_cache[key]

    def visible(key):
        seen_prefab = not transform_roots
        while key:
            seen_prefab |= key in transform_roots
            t = transforms[key]
            if not game_objects[t.m_GameObject.m_PathID].m_IsActive:
                return False
            key = t.m_Father.m_PathID
        return seen_prefab

    parts = []
    for kind in ("MeshRenderer", "SkinnedMeshRenderer"):
        for key, renderer in objects.get(kind, {}).items():
            if not renderer.m_Enabled:
                continue
            owner = renderer.m_GameObject.m_PathID
            node = transforms_by_object.get(owner)
            if node is None or not visible(node):
                continue
            mesh_filter = filters.get(owner)
            pointer = renderer.m_Mesh if kind == "SkinnedMeshRenderer" else getattr(mesh_filter, "m_Mesh", None)
            if pointer is None or not pointer.m_PathID:
                continue
            try:
                mesh = pointer.deref_parse_as_object()
            except FileNotFoundError as exc:
                if str(exc).startswith("File unity default resources not found"):
                    continue
                raise
            handler = MeshHandler(mesh)
            handler.process()
            placement = world(node)
            vertices = np.asarray(handler.m_Vertices, dtype=np.float32)
            normals = np.asarray(handler.m_Normals, dtype=np.float32) if handler.m_Normals is not None else None
            if kind == "SkinnedMeshRenderer":
                vertices, normals, placement = mesh_skin(renderer, mesh, handler, world, vertices, normals, placement)
            for index, triangles in enumerate(handler.get_triangles()):
                if not len(triangles):
                    continue
                materials = renderer.m_Materials
                material = materials[min(index, len(materials) - 1)] if materials else None
                try:
                    pointer, tint, uv, cutoff = unity_material(material)
                except FileNotFoundError as exc:
                    if not str(exc).startswith("File unity default resources not found"):
                        raise
                    pointer, tint, uv, cutoff = unity_material(None)
                texture = None
                if pointer:
                    texture = f"{prefix}:tex:{pointer.m_FileID}:{pointer.m_PathID}"
                    if not cache.has_texture(texture):
                        cache.add_texture(texture, pointer.deref_parse_as_object().image)
                name = f"{prefix}:mesh:{key}:{index}"
                cache.add_geometry(name, geometry(vertices, triangles, normals, handler.m_UV0))
                parts.append(Surface(name, placement, texture, tint, uv, cutoff))
    return parts


def asset_nodes(tree):
    yield tree
    for child in tree["children"]:
        yield from asset_nodes(child)


def assembly_surfaces(root: Path, tree, cache):
    def visit(node, parent, first=False):
        matrix = parent @ tts_transform(node["transform"], first)
        assets = {item["kind"]: item["file"] for item in node["assets"]}
        rgba = node["color"] or {}
        tint = tuple(float(rgba.get(channel, 1.0)) for channel in "rgba")
        if mesh := assets.get("mesh"):
            texture = assets.get("diffuse")
            if texture and not cache.has_texture(texture):
                cache.add_texture(texture, opened_image(root / texture))
            for part in cache.obj_parts(mesh, root / mesh):
                yield Surface(part, matrix, texture, tint,
                              cutout=0.45 if texture and node["shader"] != 0 else 0.05)
        if bundle := assets.get("bundle"):
            secondary = assets.get("bundle_secondary")
            dependencies = [root / secondary] if secondary else []
            for item in cache.bundle_parts((bundle, secondary), root / bundle, dependencies):
                yield Surface(item.geometry, matrix @ item.transform, item.texture,
                              item.color, item.uv_transform, item.cutout)
        for child in node["children"]:
            yield from visit(child, matrix)

    return list(visit(tree, identity(), True))
