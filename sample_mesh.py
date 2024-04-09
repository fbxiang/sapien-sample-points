import numpy as np
import sapien
import trimesh
from calibur import cast_graphics

from sampling import sample_surface_even


@cast_graphics
def sample2d(im: np.typing.NDArray, xy: np.typing.NDArray) -> np.typing.NDArray:
    """
    Bilinear sampling with UV coordinate in Blender convention.

    The origin ``(0, 0)`` is the bottom-left corner of the image as in most UV conventions.

    :param im: ``(H, W, ?)`` image.
    :param xy: ``(..., 2)``, should lie in ``[0, 1]`` mostly (out-of-bounds values are clamped).
    :returns: ``(..., ?)`` sampled points.
    """
    x, y = xy.x, xy.y
    x = x * im.shape[1] - 0.5
    y = y * im.shape[0] - 0.5

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0s = x0 % im.shape[1]
    x1s = x1 % im.shape[1]
    y0s = y0 % im.shape[0]
    y1s = y1 % im.shape[0]

    Ia = np.squeeze(im[y0s, x0s], axis=-2)
    Ib = np.squeeze(im[y1s, x0s], axis=-2)
    Ic = np.squeeze(im[y0s, x1s], axis=-2)
    Id = np.squeeze(im[y1s, x1s], axis=-2)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def sample_sapien_mesh(mesh: sapien.render.RenderShapeTriangleMesh, N: int):
    '''
    Args:
        mesh: sapien mesh created by sapien.render.RenderShapeTriangleMesh(filename)
        N: points to sample
    Returns:
        Dict containing all sampled numpy arrays with the following keys
           position
           normal
           base_color
           roughness
           metallic
           specular
    '''

    meshes = [
        trimesh.Trimesh(part.get_vertices(), part.get_triangles())
        for part in mesh.parts
    ]
    areas = np.array([m.area for m in meshes])
    sample_counts = (areas / areas.sum() * N).astype(np.int64)

    mat = mesh.local_pose.to_transformation_matrix()

    result = []
    for count, part in zip(sample_counts, mesh.parts):
        vertices = part.get_vertices() @ mat[:3, :3].T + mat[:3, 3]
        triangles = part.get_triangles()
        normal = part.get_vertex_normal() @ mat[:3, :3].T
        uv = part.get_vertex_uv()

        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

        points, face_indices, face_uvs = sample_surface_even(
            trimesh.Trimesh(vertices, triangles), count, seed=0
        )

        vi = triangles[face_indices]

        face_wuvs = np.concatenate(
            (1 - face_uvs[:, [0]] - face_uvs[:, [1]], face_uvs), axis=1
        )

        point_uv = (uv[vi] * face_wuvs).sum(1)
        point_normal = (normal[vi] * face_wuvs).sum(1)
        point_normal /= np.linalg.norm(point_normal, axis=-1, keepdims=True)

        # geometry normal
        # point_normal = []
        # for i, j, k in triangles[face_indices]:
        #     point_normal.append(
        #         np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i])
        #     )
        # point_normal = np.array(point_normal)
        # point_normal /= np.linalg.norm(point_normal, axis=-1, keepdims=True)

        material = part.material

        if material.diffuse_texture:
            tex = material.diffuse_texture.download()
            if tex.dtype == np.uint8:
                tex = (tex.astype(np.float64) / 255) ** 2.2
            else:
                tex = tex.astype(np.float64)
            base_color = sample2d(tex, point_uv)
        else:
            base_color = np.ones((len(points), 4)) * material.base_color

        if material.roughness_texture:
            tex = material.roughness_texture.download()
            if len(tex.shape) == 3:
                tex = tex[..., [1]]

            if tex.dtype == np.uint8:
                tex = tex.astype(np.float64) / 255
            else:
                tex = tex.astype(np.float64)

            roughness = sample2d(tex, point_uv)[..., 0]
        else:
            roughness = np.ones(len(points)) * material.roughness

        if material.metallic_texture:
            tex = material.metallic_texture.download()
            if len(tex.shape) == 3:
                tex = tex[..., [2]]

            if tex.dtype == np.uint8:
                tex = tex.astype(np.float64) / 255
            else:
                tex = tex.astype(np.float64)
            metallic = sample2d(tex, point_uv)[..., 0]
        else:
            metallic = np.ones(len(points)) * material.metallic

        specular = np.ones(len(points)) * material.specular

        result.append(
            {
                "position": points,
                "normal": point_normal,
                "base_color": base_color,
                "roughness": roughness,
                "metallic": metallic,
                "specular": specular,
            }
        )

    res = {}
    for key in result[0]:
        res[key] = np.concatenate([r[key] for r in result], 0)
    return res
