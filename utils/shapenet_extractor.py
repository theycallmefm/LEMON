import os
import math
import random
import trimesh
import pyrender
import numpy as np
import PIL
import cv2
import imageio
from tqdm import tqdm
from pathlib import Path

VIEWS=49
IMAGE_SIZE=512
FOCAL=512

Z_CENTER = 1.3
Z_RANGE = 0.2
Z_NEAR = 0.1
Z_FAR = 16

PITCH_RANGE_min, PITCH_RANGE_max = -math.pi/4, math.pi/12
YAW_RANGE_min, YAW_RANGE_max = -math.pi/6, math.pi/6
ROLL_RANGE_min, ROLL_RANGE_max = -math.pi/6, math.pi/6


class AABB:
    def __init__(self, points):
        self.min_p, self.max_p = np.amin(points, axis=0), np.amax(points, axis=0)

    @classmethod
    def load(cls, path):
        points = np.loadtxt(path)
        return cls(points.astype(np.float32))

    def save(self, path):
        np.savetxt(path, np.array(self.minmax))

    @property
    def minmax(self):
        return [self.min_p, self.max_p]

    @property
    def center(self):
        return 0.5 * (self.max_p + self.min_p)

    @property
    def longest_extent(self):
        return np.amax(self.max_p - self.min_p)

    @property
    def corners(self):
        return np.array([
            [self.min_p[0], self.min_p[1], self.min_p[2]],
            [self.max_p[0], self.min_p[1], self.min_p[2]],
            [self.max_p[0], self.max_p[1], self.min_p[2]],
            [self.min_p[0], self.max_p[1], self.min_p[2]],
            [self.min_p[0], self.min_p[1], self.max_p[2]],
            [self.max_p[0], self.min_p[1], self.max_p[2]],
            [self.max_p[0], self.max_p[1], self.max_p[2]],
            [self.min_p[0], self.max_p[1], self.max_p[2]]
        ])

def decompose(P):
    K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    c = c[:3, 0] / c[3]
    t = - R @ c
    # ensure unique scaling of K matrix
    K = K / K[2,2]
    return K, R, t

def get_colored_cube():
    side_faces = [[4,6], [10,11], [3,8], [0,2], [7,9], [1,5]]
    side_colors = [trimesh.visual.color.random_color() for _ in range(6)]
    cube = trimesh.primitives.Box()

    cube.visual.face_colors = [trimesh.visual.color.random_color()]*12
    for sf, sc in zip(side_faces, side_colors):
        cube.visual.face_colors[sf[0]] = sc
        cube.visual.face_colors[sf[1]] = sc

    return cube

def get_intrinsics():
    return pyrender.IntrinsicsCamera(
        fx=FOCAL,
        fy=FOCAL,
        cx=IMAGE_SIZE/2,
        cy=IMAGE_SIZE/2,
        znear=Z_NEAR,
        zfar=Z_FAR)

def euler_to_rotation_matrix(pitch=0.0, yaw=0.0, roll=0.0):
    R_x = np.array([[1., 0., 0.],
                    [0.,math.cos(pitch), -math.sin(pitch)],
                    [0., math.sin(pitch), math.cos(pitch)]])

    R_y = np.array([[math.cos(yaw), 0., math.sin(yaw)],
                    [0., 1., 0.],
                    [-math.sin(yaw), 0., math.cos(yaw)]])

    R_z = np.array([[math.cos(roll), -math.sin(roll), 0.],
                    [math.sin(roll), math.cos(roll), 0.],
                    [0., 0., 1.]])

    return R_z.dot(R_y).dot(R_x)

def get_random_pose(center):
    pitch = random.uniform(PITCH_RANGE_min, PITCH_RANGE_max)
    yaw = random.uniform(YAW_RANGE_min, YAW_RANGE_max)
    roll = random.uniform(ROLL_RANGE_min, ROLL_RANGE_max)
    z = random.uniform(Z_CENTER-Z_RANGE/2, Z_CENTER+Z_RANGE/2)

    Pr = np.eye(4)
    Pr[:3,:3] = euler_to_rotation_matrix(pitch, yaw, roll)

    Pt = np.eye(4)
    Pt[0,3] = center[0]
    Pt[1,3] = center[1]
    Pt[2,3] = center[2]-z

    return Pr.dot(Pt)

def to_gl(pose):
    return pose.dot(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

# Define geometry and properties
mesh_path = "./chair_gt/models/model_normalized.obj"
#mesh_path = "./scaled_garbage_bin.obj"
mesh_ = trimesh.load(mesh_path,  force='mesh')

#calculate reference position
vertices = np.array(mesh_.vertices, dtype=np.float32)
# Calculate the bounding box of the object's vertices
min_bound = np.min(vertices, axis=0)
max_bound = np.max(vertices, axis=0)
center = (min_bound + max_bound) / 2


angle = math.pi 
direction = [1, 0, 0]
rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
mesh_.apply_transform(rot_matrix)

angle = math.pi 
direction = [0, 1, 0]
rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)

mesh_.apply_transform(rot_matrix)
#mesh_.show()
#mesh_ = trimesh.load(mesh_path, process=False)
mesh = pyrender.Mesh.from_trimesh(mesh_, smooth=False)





# Define light
lights = [pyrender.PointLight(intensity=50.0) for _ in range(VIEWS)]
light_poses = [to_gl(get_random_pose(center)) for _ in range(VIEWS)]

# Define camera
intrinsics = get_intrinsics()

images = []
masks = []
poses = []
for view in range(VIEWS):

    # Define scene
    scene = pyrender.Scene()
    scene.add(mesh)
    for l, lp in zip(lights, light_poses): scene.add(l, pose=lp)
    pose = get_random_pose(center)
    scene.add(intrinsics, pose=to_gl(pose))
    poses.append(pose)

    # Render
    renderer = pyrender.OffscreenRenderer(IMAGE_SIZE, IMAGE_SIZE)
    image, depth = renderer.render(scene)
    images.append(image)
    masks.append(depth != 0.)

# Export
data_dummy_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dummy', 'scan0')
print(data_dummy_dir)

# Export cameras
cameras = {}
k = np.eye(3)
k[:2,:2] *= FOCAL
k[:2,2] = IMAGE_SIZE/2
for idx, p in enumerate(poses):
    p = np.linalg.inv(p)

    wm = np.eye(4)
    wm[:3,:3] = k@p[:3,:3]
    wm[:3,3] = k@p[:3,3]

    cameras['scale_mat_%d' % idx] = np.eye(4)
    cameras['world_mat_%d' % idx] = wm
np.savez(os.path.join(data_dummy_dir,'cameras.npz'), **cameras)

# Export images and masks
data_dummy_image_dir = os.path.join(data_dummy_dir, 'image')
data_dummy_mask_dir = os.path.join(data_dummy_dir, 'mask')
if not os.path.exists(data_dummy_dir): os.mkdir(data_dummy_dir)
if not os.path.exists(data_dummy_image_dir): os.mkdir(data_dummy_image_dir)
if not os.path.exists(data_dummy_mask_dir): os.mkdir(data_dummy_mask_dir)
for idx, (i, m) in enumerate(zip(images, masks)):
    PIL.Image.fromarray(i.astype(np.uint8)).save(os.path.join(data_dummy_image_dir, 'img_{0:04}.png'.format(idx)))
    PIL.Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(data_dummy_mask_dir, 'mask_{0:04}.png'.format(idx)))


input_dir = Path("./dummy")
output_dir = Path("./dummy_chair")
for scan_dir in [directory for directory in input_dir.iterdir() if directory.is_dir()]:
        print(f"-- Converting {scan_dir.name}")

        scan_output_dir = output_dir / scan_dir.name

        scan_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect the image and mask paths
        image_dir = scan_dir / 'image'
        mask_dir  = scan_dir / 'mask'

        image_paths = sorted([path for path in image_dir.iterdir() if (path.is_file() and path.suffix == '.png' and not str(path.stem).startswith('.'))])
        mask_paths  = sorted([path for path in mask_dir.iterdir() if (path.is_file() and path.suffix == '.png' and not str(path.stem).startswith('.'))])

        # Read cameras
        cameras = np.load(scan_dir / 'cameras.npz')

        # Make a unit aabb 
        #! WARNING! This assumes a symmetric bbox (aka cube) if the bbox is not a cube 
        #! this will not give the bbox back, but something similar
        bbox = np.array([[-.5, -.5, -.5],
                         [.5,  .5,  .5]], dtype=np.float32)

        views_output_dir = scan_output_dir / "views"
        views_output_dir.mkdir(parents=True, exist_ok=True)
        for i, view in enumerate(image_paths):
            K, R, t = decompose(cameras[f"world_mat_{i}"][:3,:])
            np.savetxt(views_output_dir / f'cam{i:06d}_r.txt', R, fmt='%.20f')
            np.savetxt(views_output_dir / f'cam{i:06d}_t.txt', t, fmt='%.20f')
            np.savetxt(views_output_dir / f'cam{i:06d}_k.txt', K, fmt='%.20f')

            A_inv = cameras[f"scale_mat_{i}"]
            bbox_denormalized = (bbox @ A_inv[:3, :3].T) + A_inv[:3, 3][np.newaxis, :]

        # Save the bounding box
        # NOTE (MW): This assumes that all scale_mats are equal for a scan (which seems to be the case)
        #            Otherwise we would choose the last one, which seems a bit arbitrary
        aabb = AABB(bbox_denormalized)
        aabb.save(scan_output_dir / "bbox.txt")
        
        # Convert the images and masks to a single RGBA PNG
        for idx, (image, mask) in tqdm(enumerate(zip(image_paths, mask_paths)), leave=False):
            color = np.concatenate((imageio.imread(image),
                                    imageio.imread(mask, pilmode='L')[..., np.newaxis]), 
                                    axis=-1)
            imageio.imwrite(views_output_dir / f'cam{idx:06d}.png', color)