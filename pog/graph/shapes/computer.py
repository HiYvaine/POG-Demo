import numpy as np
import sdf.d2
import trimesh
from pog.graph.shape import Affordance, Shape, ShapeID, ShapeType
from trimesh.transformations import rotation_matrix, translation_matrix
from numpy import pi as PI
from pog.graph.shapes.complex_storage import compute_door_swept_concatenate
from pog.graph.node import ContainmentState

keyboard_height_ratio = 2/3.0

class Computer(Shape):

    def __init__(self,
                 shape_type=ShapeID.Computer,
                 size=np.array([0.32, 0.22, 0.015]),
                 transform=np.identity(4),
                 state = 0,
                 joint_dmax=PI/1.8,
                 **kwargs) -> None:

        super().__init__(shape_type)
        self.size = np.array(size)

        if   state == 0: self.state = ContainmentState.Closed
        elif state == 1: self.state = ContainmentState.Opened
        else: raise Exception('Invalid state of ComplexStorage')
   
        computer_color = np.array([200, 200, 200, 255], dtype=np.uint8) 
        keyboard_size = np.array([size[0], size[1], size[2]*keyboard_height_ratio])
        screen_size = np.array([size[0], size[1], size[2]*(1.0-keyboard_height_ratio)])

        # no transform, origin is set to keyboard center
        keyboard_shape = trimesh.creation.box(
                extents=keyboard_size,
                transform=np.identity(4),
                **kwargs,
            )        
        keyboard_shape.visual.face_colors[:] = computer_color
        screen_shape = trimesh.creation.box(
                extents=screen_size,
                transform=translation_matrix([0, 0, self.size[2]/2.0]),
                **kwargs,
            )
        screen_shape.visual.face_colors[:] = computer_color

        self.shape: trimesh.Trimesh = trimesh.util.concatenate([keyboard_shape, screen_shape])
        self.shape.apply_transform(transform)
        self.close_shape = self.shape.copy()

        rotation_center = [0, -keyboard_size[1]/2, keyboard_size[2]/2]
        rotation_axis = [1, 0, 0]
        end_angle = joint_dmax
        combined_transform = rotation_matrix(end_angle, rotation_axis, point=rotation_center)
        screen_open_shape = screen_shape.copy()
        screen_open_shape.apply_transform(combined_transform)

        self.open_shape: trimesh.Trimesh = trimesh.util.concatenate([keyboard_shape, screen_open_shape])
        self.open_shape.apply_transform(transform)

        screen_swept_shape = compute_door_swept_concatenate(screen_shape, rotation_center, rotation_axis, end_angle, interpolations=3)
        self.open_swept_shape: trimesh.Trimesh = trimesh.util.concatenate(keyboard_shape, screen_swept_shape)
        self.open_swept_shape.apply_transform(transform)
        self.close_swept_shape = self.open_swept_shape.copy()

        self.transform = transform
        self.volume = self.size[0] * self.size[1] * self.size[2]
        self.mass = self.volume
        self.object_type = ShapeType.ARTIC
        self.com = np.array(self.shape.center_mass)
        self.create_aff()

    @property
    def SHAPE_TYPE(self):
        return ShapeID.Computer

    @classmethod
    def from_saved(cls, n: dict):
        params = {
            "size": n["size"],
            "transform": np.array(n["transform"]),
        }
        attr_names = ["joint_dmax", "state"]
        for attr in attr_names:
            if attr in n:
                params[attr] = n[attr]

        return cls(**params)

    def create_aff(self):
        keyboard_height = self.size[2] * keyboard_height_ratio
        # print(type(self.transform @ translation_matrix([0, 0, keyboard_height/2.])))
        tf = self.transform @ translation_matrix([0, 0, keyboard_height/2.])
        print(type(tf))
        aff_pz = Affordance(
            name='keyboard_support', 
            transform=tf, 
            containment = True,
            shape = sdf.d2.rectangle(self.size[[0,1]]), 
            area = self.size[0]*self.size[1], 
            bb = self.size[[0,1]], 
            height = self.size[1])


        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, -keyboard_height/2.0),
             (0, 0, 0, 1)))
        aff_nz = Affordance(
            name='keyboard_stand', 
            transform=tf, 
            containment = False,
            shape = sdf.d2.rectangle(self.size[[0,1]]), 
            area = self.size[0]*self.size[1], 
            bb = self.size[[0,1]], 
            height = self.size[2])

        self.add_aff(aff_pz)
        self.add_aff(aff_nz)