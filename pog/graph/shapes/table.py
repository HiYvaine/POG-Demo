import numpy as np
import sdf.d2
import trimesh
from pog.graph.shape import Affordance, Shape, ShapeID

Tabletop_Thickness = 0.02
Tableleg_Thickness = 0.03
Tableleg_Inset = 0.04

class Table(Shape):

    def __init__(self,
                 shape_type=ShapeID.Table,
                 size=np.array([0.8, 0.8, 0.4]),
                 with_shelf = False,
                 shelf_size = np.array([0.6, 0.6, 0.2]),
                 transform=np.identity(4),
                 **kwargs) -> None:
        """Basic box shape

        Args:
            shape_type (ShapeID, optional): shape type. Defaults to ShapeID.Box.
            size (float or (3,) float, optional): size of box. Defaults to 1.0.
            transform (4x4 numpy array, optional): transformation of this shape. Defaults to np.identity(4).
        """
        super().__init__(shape_type)
        self.size = np.array(size)
        self.with_shelf = with_shelf
        self.shelf_size = np.array(shelf_size)

        self.shape = self.create_table(size=self.size, with_shelf=self.with_shelf, shelf_size=self.shelf_size)
        self.shape.visual.face_colors[:] = trimesh.visual.random_color()
        self.transform = transform
        self.volume = self.size[0] * self.size[1] * self.size[2]
        self.mass = self.volume
        self.com = np.array(self.shape.center_mass)
        self.create_aff()
        self.export_obj()

    @classmethod
    def from_saved(cls, n: dict):
        return cls(size=n['size'], transform=np.array(n['transform']))

    @property
    def export_file_name(self):
        return '/home/user/POG-Demo/pog_example/mesh/temp_box_{:.1f}_{:.1f}_{:.1f}.obj'.format(
            self.size[0], self.size[1], self.size[2])

    def create_aff(self):
        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, self.size[2] / 2.0),
             (0, 0, 0, 1)))
        aff_pz = Affordance(name='table_surface', transform=tf, containment = False,\
            shape = sdf.d2.rectangle(self.size[[0,1]]), area = self.size[0]*self.size[1], bb = self.size[[0,1]], height = self.size[2])


        tf = self.transform @ np.array(
            ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, -self.size[2] / 2.0),
             (0, 0, 0, 1)))
        aff_nz = Affordance(name='table_stand', transform=tf, containment = False,\
            shape = sdf.d2.rectangle(self.size[[0,1]]), area = self.size[0]*self.size[1], bb = self.size[[0,1]], height = self.size[2])

        self.add_aff(aff_pz)
        self.add_aff(aff_nz)


    @property
    def default_affordance_name(self) -> str:
        return 'table_surface'
    
    @staticmethod
    def create_table(size, with_shelf=False, shelf_size=None):
        """
        Create a table as a trimesh shape.

        Args:
            size (list or np.array): Overall dimensions of the table [length, width, height].
            Tabletop_Thickness (float): Thickness of the tabletop.
            Tableleg_Thickness (float): Cross-sectional size of the table legs (square).
            Tableleg_Inset (float): Inset distance of the table legs from the edges of the tabletop.

        Returns:
            trimesh.Trimesh: The combined mesh of the table.
        """
        tabletop = trimesh.creation.box(
            extents=[size[0], size[1], Tabletop_Thickness],
            transform=trimesh.transformations.translation_matrix(
                [0, 0, size[2] / 2 - Tabletop_Thickness / 2]
            )
        )

        leg_height = size[2] - Tabletop_Thickness
        leg_positions = [
            [size[0] / 2 - Tableleg_Inset - Tableleg_Thickness / 2,
            size[1] / 2 - Tableleg_Inset - Tableleg_Thickness / 2,
            leg_height / 2 - size[2] / 2],
            [-size[0] / 2 + Tableleg_Inset + Tableleg_Thickness / 2,
            size[1] / 2 - Tableleg_Inset - Tableleg_Thickness / 2,
            leg_height / 2 - size[2] / 2],
            [size[0] / 2 - Tableleg_Inset - Tableleg_Thickness / 2,
            -size[1] / 2 + Tableleg_Inset + Tableleg_Thickness / 2,
            leg_height / 2 - size[2] / 2],
            [-size[0] / 2 + Tableleg_Inset + Tableleg_Thickness / 2,
            -size[1] / 2 + Tableleg_Inset + Tableleg_Thickness / 2,
            leg_height / 2 - size[2] / 2],
        ]

        legs = []
        for pos in leg_positions:
            leg = trimesh.creation.box(
                extents=[Tableleg_Thickness, Tableleg_Thickness, leg_height],
                transform=trimesh.transformations.translation_matrix(pos)
            )
            legs.append(leg)

        table = trimesh.util.concatenate([tabletop] + legs)

        if with_shelf and shelf_size is not None:
            shelf_panels = [
                trimesh.creation.box(
                    extents=[Tabletop_Thickness, shelf_size[1], shelf_size[2]],
                    transform=trimesh.transformations.translation_matrix(
                        [shelf_size[0] / 2 - Tabletop_Thickness / 2, 0,
                        size[2] / 2 - Tabletop_Thickness - shelf_size[2] / 2]
                    )
                ),
                trimesh.creation.box(
                    extents=[Tabletop_Thickness, shelf_size[1], shelf_size[2]],
                    transform=trimesh.transformations.translation_matrix(
                        [-shelf_size[0] / 2 + Tabletop_Thickness / 2, 0,
                        size[2] / 2 - Tabletop_Thickness - shelf_size[2] / 2]
                    )
                ),
                trimesh.creation.box(
                    extents=[shelf_size[0], shelf_size[1], Tabletop_Thickness],
                    transform=trimesh.transformations.translation_matrix(
                        [0, 0, size[2] / 2 - Tabletop_Thickness - shelf_size[2] + Tabletop_Thickness / 2]
                    )
                )
            ]
            table_shelf: trimesh.Trimesh = trimesh.util.concatenate(shelf_panels)

            shelf_transform = trimesh.transformations.translation_matrix(
                [0, size[1] / 2 - shelf_size[1] / 2, 0])
            table_shelf.apply_transform(shelf_transform)
            table = trimesh.util.concatenate(table, table_shelf)
            table.show()

        return table