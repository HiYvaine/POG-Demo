from array import ArrayType

import numpy as np
import pybullet as p
import sdf.d2
import sdf.d3
import transforms3d
import trimesh
import trimesh.creation as creation
from pog.graph.node import ContainmentState
from pog.graph.params import BULLET_GROUND_OFFSET, WALL_THICKNESS
from pog.graph.shape import Affordance, Shape, ShapeID, ShapeType
from trimesh.transformations import rotation_matrix
from scipy.spatial import ConvexHull
from trimesh.transformations import translation_matrix

# Don't use — in_collision_internal can't detect full containment.
def compute_door_swept_convex_hull(door_shape, rotation_center, rotation_axis, swept_angle, transform=None, interpolations=3):
# Apply [transform] if doo_shape has been transformed in global but rotation params are in local.
    vertices = []
    angles = np.linspace(0, swept_angle, interpolations+2)
    transform = transform if transform is not None else np.identity(4)
    for angle in angles:
        interp_door = door_shape.copy()
        local_transform = rotation_matrix(angle, rotation_axis, point=rotation_center)
        interp_door.apply_transform(transform @ local_transform)
        vertices.append(interp_door.vertices)
    vertices = np.vstack(vertices)
    hull = ConvexHull(vertices)
    swept_mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices)
    return swept_mesh

def compute_door_swept_concatenate(door_shape, rotation_center, rotation_axis, swept_angle, transform=None, interpolations=6):
# Apply [transform] if doo_shape has been transformed in global but rotation params are in local.
    interp_doors = []
    angles = np.linspace(0, swept_angle, interpolations+2)
    transform = transform if transform is not None else np.identity(4)
    for angle in angles:
        interp_door = door_shape.copy()
        local_transform = rotation_matrix(angle, rotation_axis, point=rotation_center)
        interp_door.apply_transform(transform @ local_transform)
        interp_doors.append(interp_door)
    swept_mesh: trimesh.Trimesh = trimesh.util.concatenate(interp_doors)
    return swept_mesh

class ComplexStorage(Shape):
    size: ArrayType

    def __init__(self,
                 shape_type=ShapeID.ComplexStorage,
                 size=np.array([0.8, 0.8, 1.0]),
                 transform=np.identity(4),
                 storage_type='cabinet',
                 with_door=True,
                 with_board=True,
                 joint_axis='x',
                 joint_dmax=np.pi/2,
                 state=0,
                 **kwargs):
        """
        size: size in xyz
        """
        super().__init__(shape_type)
        if shape_type != self.SHAPE_TYPE:
            raise Exception(
                "invalid shape type of complex store: {}".format(shape_type))
        size = np.array(size)
        self.size = size
        self.with_door = with_door

        if   state == 0: self.state = ContainmentState.Closed
        elif state == 1: self.state = ContainmentState.Opened
        else: raise Exception('Invalid state of ComplexStorage')
        
        # no transform, the ComplexStorage opens by default along the positive Y-axis. 
        outer_shape = creation.box(size, np.identity(4), **kwargs)
        inner_shape = creation.box(
            extents=[size[0]-WALL_THICKNESS*2, size[1]-WALL_THICKNESS, size[2]-WALL_THICKNESS*2],
            transform=translation_matrix([0, WALL_THICKNESS/2., 0]),
            **kwargs,
        )
        body_shape: trimesh.Trimesh = outer_shape.difference(inner_shape)
        self.with_board = with_board
        if self.with_board:         
            board_shape = creation.box(
                extents=[size[0]-WALL_THICKNESS*2, size[1]-WALL_THICKNESS*2, WALL_THICKNESS],
                transform=np.identity(4),
                **kwargs,
            )                         
            body_shape: trimesh.Trimesh = trimesh.util.concatenate(body_shape, board_shape)

        body_color = trimesh.visual.random_color()
        body_shape.visual.face_colors = body_color
        self.shape = body_shape.copy()
        self.shape.apply_transform(transform)

        if with_door:
            door_shape = creation.box(
                extents=[size[0]-WALL_THICKNESS*2, WALL_THICKNESS, size[2]-WALL_THICKNESS*2],
                transform=translation_matrix([0, (size[1]-WALL_THICKNESS)/2., 0]),
                **kwargs
                )
            door_color = body_color.copy()
            door_color[3] = 200
            door_shape.visual.face_colors = door_color

            self.close_shape: trimesh.Trimesh = trimesh.util.concatenate(body_shape, door_shape)
            self.close_shape.apply_transform(transform)
            door_open_shape = door_shape.copy()

            if joint_axis == 'z':
                rotation_center = [size[0]/2-WALL_THICKNESS, size[1]/2, 0]
                rotation_axis = [0, 0, 1]
                end_angle = -joint_dmax

            elif joint_axis == 'z-right':
                rotation_center = [-size[0]/2+WALL_THICKNESS, size[1]/2, 0]
                rotation_axis = [0, 0, 1]
                end_angle = joint_dmax

            elif joint_axis == 'x':
                rotation_center = [0, size[1]/2, size[2]/2-WALL_THICKNESS]
                rotation_axis = [1, 0, 0]
                end_angle = joint_dmax

            elif joint_axis == 'x-bottom':
                rotation_center = [0, size[1]/2, -size[2]/2+WALL_THICKNESS]
                rotation_axis = [1, 0, 0]
                end_angle = -joint_dmax
            
            else:
                raise ValueError(
                    f"Invalid joint_axis '{joint_axis}'; expected: 'z', 'z-right', 'x', 'x-bottom'."
                )

            local_transform = rotation_matrix(end_angle, rotation_axis, point=rotation_center)
            door_open_shape.apply_transform(local_transform)
            self.open_shape: trimesh.Trimesh = trimesh.util.concatenate(body_shape, door_open_shape)
            self.open_shape.apply_transform(transform)
            door_swept_shape = compute_door_swept_concatenate(door_shape, rotation_center, rotation_axis, end_angle)
            self.open_swept_shape: trimesh.Trimesh = trimesh.util.concatenate(body_shape, door_swept_shape)
            self.open_swept_shape.apply_transform(transform)
            self.close_swept_shape = self.open_swept_shape.copy()

        self.transform = transform
        self.volume = self.shape.volume
        self.mass = self.volume
        self.object_type = ShapeType.ARTIC
        self.com = np.array(self.shape.center_mass)
        self.create_aff(storage_type, size)

    @property
    def SHAPE_TYPE(self):
        return ShapeID.ComplexStorage

    @classmethod
    def from_saved(cls, n: dict):
        params = {
            "size": n["size"],
            "transform": np.array(n["transform"]),
        }
        if "with_door" in n:
            params["with_door"] = n["with_door"]
        if "joint_axis" in n:
            params["joint_axis"] = n["joint_axis"]
        if "joint_dmax" in n:
            params["joint_dmax"] = n["joint_dmax"]
        if "state" in n:
            params["state"] = n["state"]
        if "with_board" in n:
            params["with_board"] = n["with_board"]

        return cls(**params)

    def create_aff(self, storage_type: str, size):
        outer_params = {
            "containment": False,
            "shape": sdf.d2.rectangle(size[[0, 1]]),
            "area": size[0] * size[1],
            "bb": size[[0, 1]],
            "height": size[2],
        }

        length_x = size[0] - WALL_THICKNESS * 2.
        length_y = size[1] - WALL_THICKNESS * 2.
        height = size[2] - WALL_THICKNESS * 2.
        if self.with_board:
            height = (size[2] - WALL_THICKNESS * 3.) / 2.

        inner_params = {
            "containment": self.with_door,
            "shape": sdf.d2.rectangle([length_x, length_y]),
            "area": length_x * length_y,
            "bb": [length_x, length_y],
            "height": height,
        }
        if storage_type == 'cabinet':
            aff_dicts = self.get_cabinet_affs(inner_params, outer_params)
            for aff in aff_dicts:
                self.add_aff(
                    Affordance(name=aff["name"],
                               transform=aff["tf"],
                               **aff["params"]))

    def get_cabinet_affs(self, inner_params, outer_params):
        cabinet_affs = [{
            "name": 'cabinet_outer_top',
            "tf": self.transform @ translation_matrix([0, 0, self.size[2]/2.0]),
            "params": outer_params,
        },{
            "name": "cabinet_outer_bottom",
            "tf": self.transform @ np.array((
                (1, 0, 0, 0),
                (0, -1, 0, 0),
                (0, 0, -1, -self.size[2]/2.0),
                (0, 0, 0, 1),
            )),
            "params": outer_params,
        },{
            "name": "cabinet_inner_bottom",
            "tf": self.transform @ translation_matrix([0, 0, -self.size[2]/2.0+WALL_THICKNESS]),
            "params": inner_params,
            }]
        if self.with_board:
            cabinet_affs.append({
            "name":
            "cabinet_inner_middle",
            "tf": self.transform @ translation_matrix([0, 0, WALL_THICKNESS/2.0]),
            "params": inner_params,
        })
        return cabinet_affs

    def create_bullet_shapes(self, global_transform):
        visual_shapes = []
        collision_shapes = []
        halfwlExtents = [
            self.size[0] / 2., self.size[1] / 2., WALL_THICKNESS / 4.
        ]
        halflhExtents = [
            self.size[0] / 2., WALL_THICKNESS / 4., self.size[2] / 2.
        ]
        halfwhExtents = [
            WALL_THICKNESS / 4., self.size[1] / 2., self.size[2] / 2.
        ]
        shape_params = [{
            "ext":
            halfwlExtents,
            "frame_position": [0, 0, -self.size[2] / 2. + WALL_THICKNESS / 4.]
        }, {
            "ext": halfwlExtents
        }, {
            "ext": halfwlExtents
        }, {
            "ext": halfwhExtents
        }, {
            "ext": halfwhExtents
        }, {
            "ext": halflhExtents
        }, {
            "ext":
            halflhExtents,
            "frame_position": [-self.size[0] / 2. + WALL_THICKNESS / 4., 0, 0]
        }]
        for param in shape_params:
            visual_shapes.append(
                p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=param["ext"],
                    visualFramePosition=param.get("frame_position", None),
                ))
            collision_shapes.append(
                p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=param["ext"],
                    # halfExtents=[0, 0, 0],
                    collisionFramePosition=param.get("frame_position", None),
                ))

        visual_shapes.append(
            p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.015))
        collision_shapes.append(
            p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.001))

        translation, rotation, _, _ = transforms3d.affines.decompose44(
            global_transform)

        quaternion = transforms3d.quaternions.mat2quat(rotation)

        if translation[-1] < 0:
            rotx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            translation = rotx180 @ translation
            rotation = rotx180 @ rotation
            quaternion = transforms3d.quaternions.mat2quat(rotation)
        multibody = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shapes[0],
            baseVisualShapeIndex=visual_shapes[0],
            basePosition=translation + BULLET_GROUND_OFFSET,
            baseOrientation=np.hstack((quaternion[1:4], quaternion[0])),
            linkMasses=[0, 0, 0, 0, 0, 1, 0],
            linkCollisionShapeIndices=collision_shapes[1:],
            linkVisualShapeIndices=visual_shapes[1:],
            linkPositions=[
                [0, 0, self.size[2] / 2. - WALL_THICKNESS / 4.],
                [0, 0, 0],
                [self.size[0] / 2. - WALL_THICKNESS / 4., 0, 0],
                [-self.size[0] / 2. - WALL_THICKNESS / 4., 0, 0],
                [0, -self.size[1] / 2. + WALL_THICKNESS / 4., 0],
                [
                    self.size[0] / 2. - WALL_THICKNESS / 4.,
                    self.size[1] / 2. + WALL_THICKNESS / 4., 0
                ],
                [-self.size[0] + 0.1, 0, 0],
            ],
            linkOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkInertialFramePositions=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            linkInertialFrameOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkParentIndices=[0, 0, 0, 0, 0, 0, 6],
            linkJointTypes=[
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_REVOLUTE,
                p.JOINT_FIXED,
            ],  # related to door of storage
            linkJointAxis=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],  # also joint related
            useMaximalCoordinates=False)
        p.changeDynamics(multibody,
                         5,
                         jointLowerLimit=-3.14,
                         jointUpperLimit=0)
        p.changeVisualShape(multibody,
                            linkIndex=4,
                            rgbaColor=[0.76470588, 0.765, 0.765, 1.],
                            specularColor=[0.4, 0.4, 0])
        return visual_shapes, collision_shapes, multibody
