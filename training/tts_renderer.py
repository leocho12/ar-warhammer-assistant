"""Off-screen rendering and instance-mask annotation for tabletop miniatures."""

from dataclasses import dataclass
from itertools import product

import cv2
import moderngl
import numpy as np
from PIL import Image

from tts_render_assets import euler, geometry, identity, obj_geometry, scale, translation, unity_bundle


VERTEX_SHADER = """
#version 330
in vec3 position;
in vec3 normal;
in vec2 uv;
uniform mat4 model;
uniform mat4 view_projection;
out vec3 world;
out vec3 direction;
out vec2 texcoord;
void main() {
    vec4 point = model * vec4(position, 1.0);
    world = point.xyz;
    direction = transpose(inverse(mat3(model))) * normal;
    texcoord = uv;
    gl_Position = view_projection * point;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 world;
in vec3 direction;
in vec2 texcoord;
layout(location = 0) out vec4 color;
layout(location = 1) out float instance_mask;
uniform sampler2D albedo;
uniform vec4 tint;
uniform vec4 uv_transform;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform vec3 tabletop;
uniform vec3 eye;
uniform vec4 shadows[8];
uniform float cutout;
uniform float style;
uniform int object_id;
uniform int shadow_count;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

void main() {
    if (object_id == 0) {
        vec2 p = world.xz;
        float texture = 0.72 + 0.22 * noise(p*3.6) + 0.13 * noise(p*19.0) + 0.13 * noise(p*88.0);
        if (style > 1.5) {
            vec2 tile = abs(fract(p*0.95 + 0.35) - 0.5);
            texture -= 0.21 * smoothstep(0.476, 0.496, max(tile.x, tile.y));
        } else if (style > 0.5) {
            texture += 0.11 * noise(vec2(p.x*96.0, p.y*5.0));
        }
        float shade = 1.0;
        for (int i=0; i<shadow_count; i++) {
            vec2 d = (p - shadows[i].xy) / shadows[i].zw;
            shade *= 1.0 - 0.50 * exp(-2.0 * dot(d, d));
        }
        color = vec4(tabletop * texture * shade, 1.0);
        instance_mask = 0.0;
    } else {
        vec4 material = texture(albedo, texcoord * uv_transform.xy + uv_transform.zw) * tint;
        if (material.a < cutout) discard;
        vec3 n = normalize(direction + vec3(0.000001));
        float direct = max(0.0, dot(n, light_direction));
        float sky = 0.5 * n.y + 0.5;
        vec3 h = normalize(normalize(eye - world) + light_direction);
        float highlight = 0.07 * pow(max(dot(n, h), 0.0), 24.0) * direct;
        color = vec4(material.rgb * (0.43 + 0.18*sky + light_color*direct) + highlight, 1.0);
        instance_mask = float(object_id) / 255.0;
    }
}
"""

TABLETOPS = np.array([
    [0.29, 0.34, 0.26], [0.35, 0.32, 0.28], [0.49, 0.43, 0.35],
    [0.24, 0.26, 0.30], [0.54, 0.52, 0.49], [0.33, 0.39, 0.39],
    [0.37, 0.27, 0.23], [0.40, 0.38, 0.27], [0.22, 0.21, 0.22],
], dtype=np.float32)


@dataclass
class GpuMesh:
    vertex_buffer: object
    index_buffer: object
    array: object
    bounds: np.ndarray

    def release(self):
        self.array.release()
        self.vertex_buffer.release()
        self.index_buffer.release()


@dataclass
class Subject:
    sample: dict
    surfaces: list
    transform: np.ndarray
    dimensions: np.ndarray
    original_extent: float


def unit(vector):
    return vector / np.linalg.norm(vector)


def view_projection(eye, target, fov):
    forward = unit(target - eye)
    right = unit(np.cross(forward, [0, 1, 0]))
    upward = np.cross(right, forward)
    view = identity()
    view[:3, :3] = np.array([right, upward, -forward])
    view[:3, 3] = -(view[:3, :3] @ eye)
    near, far = 0.04, 100.0
    focal = 1 / np.tan(np.deg2rad(fov) / 2)
    projection = np.array(
        [[focal, 0, 0, 0], [0, focal, 0, 0], [0, 0, -(far + near) / (far - near),
                                        -2 * near * far / (far - near)], [0, 0, -1, 0]],
        dtype=np.float32,
    )
    return projection @ view


def write_matrix(uniform, matrix):
    uniform.write(np.asarray(matrix.T, dtype=np.float32).tobytes())


def camera(target, azimuth, elevation, distance):
    theta, phi = np.deg2rad([azimuth, elevation])
    return target + np.array([np.cos(theta)*np.cos(phi), np.sin(phi), np.sin(theta)*np.cos(phi)]) * distance


class Renderer:
    def __init__(self, size, bundle_library=None):
        self.size = size
        self.bundle_library = bundle_library or {}
        self.context = moderngl.create_standalone_context(require=330)
        self.context.enable(moderngl.DEPTH_TEST)
        self.program = self.context.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.color = self.context.texture((size, size), 4)
        self.mask = self.context.texture((size, size), 1)
        self.depth = self.context.depth_renderbuffer((size, size))
        self.frame = self.context.framebuffer((self.color, self.mask), self.depth)
        self.meshes = {}
        self.textures = {}
        self.asset_surfaces = {}
        self.white = self.context.texture((1, 1), 4, b"\xff" * 4)
        vertices = [(-40, -0.003, -40), (-40, -0.003, 40), (40, -0.003, 40), (40, -0.003, -40)]
        self.floor = self.upload(geometry(vertices, [(0, 1, 2), (0, 2, 3)], [(0, 1, 0)] * 4, None))

    def upload(self, source):
        vbo = self.context.buffer(source.vertices.tobytes())
        ibo = self.context.buffer(source.faces.astype(np.uint32, copy=False).tobytes())
        array = self.context.vertex_array(self.program, [(vbo, "3f 3f 2f", "position", "normal", "uv")], ibo)
        return GpuMesh(vbo, ibo, array, source.bounds)

    def add_geometry(self, name, source):
        self.meshes[name] = self.upload(source)

    def has_texture(self, name):
        return name in self.textures

    def add_texture(self, name, image):
        image = image.convert("RGBA")
        image.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
        data = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes()
        texture = self.context.texture(image.size, 4, data)
        texture.build_mipmaps()
        self.textures[name] = texture

    def obj_parts(self, identifier, path):
        if identifier not in self.asset_surfaces:
            parts = []
            for index, mesh in enumerate(obj_geometry(path)):
                key = f"{identifier}:{index}"
                self.add_geometry(key, mesh)
                parts.append(key)
            if not parts:
                raise ValueError("OBJ contains no renderable triangles")
            self.asset_surfaces[identifier] = parts
        return self.asset_surfaces[identifier]

    def bundle_parts(self, identifier, path, dependencies):
        if identifier not in self.asset_surfaces:
            prefix = "::".join(item for item in identifier if item)
            self.asset_surfaces[identifier] = unity_bundle(path, dependencies, self, prefix)
        return self.asset_surfaces[identifier]

    def clear_assets(self):
        for mesh in self.meshes.values():
            mesh.release()
        for texture in self.textures.values():
            texture.release()
        self.asset_surfaces.clear()
        self.meshes.clear()
        self.textures.clear()

    def subject(self, sample, surfaces):
        if not surfaces:
            raise ValueError("Assembly contains no surfaces")
        corners = []
        for surface in surfaces:
            bounds = self.meshes[surface.geometry].bounds
            points = np.asarray(list(product(*bounds.T)), dtype=np.float32)
            corners.extend(points @ surface.transform[:3, :3].T + surface.transform[:3, 3])
        bounds = np.asarray(corners)
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        span = high - low
        extent = float(span.max())
        if not np.isfinite(extent) or extent < 1.0e-6:
            raise ValueError("Assembly has empty or invalid bounds")
        cx, _, cz = (low + high) / 2
        normalization = scale(*(np.ones(3) / extent)) @ translation(-cx, -low[1], -cz)
        return Subject(sample, surfaces, normalization, span / extent, extent)

    def render(self, subjects, target, azimuth, elevation, distance, fov, rng):
        eye = camera(target, azimuth, elevation, distance)
        self.frame.use()
        self.frame.clear(0, 0, 0, 0, depth=1)
        p = self.program
        write_matrix(p["view_projection"], view_projection(eye, target, fov))
        p["eye"].value = tuple(eye)
        light = unit(np.array([rng.uniform(-1.5, 1.5), rng.uniform(0.55, 2), rng.uniform(-1.5, 1.5)]))
        p["light_direction"].value = tuple(light)
        p["light_color"].value = tuple(rng.uniform(0.38, 0.66, size=3))
        p["tabletop"].value = tuple(TABLETOPS[rng.integers(len(TABLETOPS))] * rng.uniform(0.80, 1.18))
        p["style"].value = float(rng.integers(0, 3))
        shadows = np.zeros((8, 4), dtype=np.float32)
        for i, (subject, position, angle, multiplier) in enumerate(subjects):
            x, z = position
            footprint = max(subject.dimensions[0], subject.dimensions[2], 0.36) * multiplier
            shadows[i] = [x - light[0]*0.09, z - light[2]*0.09, footprint*0.70, footprint*0.70]
        p["shadows"].write(shadows.tobytes())
        p["shadow_count"].value = len(subjects)
        p["object_id"].value = 0
        write_matrix(p["model"], identity())
        self.white.use()
        self.floor.array.render()
        for instance, (subject, position, angle, multiplier) in enumerate(subjects, 1):
            p["object_id"].value = instance
            model = translation(position[0], 0, position[1]) @ euler(0, angle, 0)
            model = model @ scale(multiplier, multiplier, multiplier) @ subject.transform
            for surface in subject.surfaces:
                write_matrix(p["model"], model @ surface.transform)
                p["tint"].value = surface.color
                p["uv_transform"].value = surface.uv_transform
                p["cutout"].value = surface.cutout
                self.textures.get(surface.texture, self.white).use()
                self.meshes[surface.geometry].array.render()
        mask = np.frombuffer(self.mask.read(alignment=1), dtype=np.uint8).reshape(self.size, self.size)[::-1]
        boxes = []
        for instance, (subject, *_unused) in enumerate(subjects, 1):
            visible = cv2.compare(mask, instance, cv2.CMP_EQ)
            if cv2.countNonZero(visible) < 64:
                continue
            x, y, width, height = cv2.boundingRect(visible)
            boxes.append({
                "label": subject.sample["label"], "assembly": subject.sample["assembly"],
                "geometry_group": subject.sample["group"], "xyxy": [x, y, x+width, y+height],
            })
        if subjects and len(boxes) != len(subjects):
            return None, []
        rgba = np.frombuffer(self.color.read(alignment=1), dtype=np.uint8).reshape(self.size, self.size, 4)[::-1]
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), boxes
