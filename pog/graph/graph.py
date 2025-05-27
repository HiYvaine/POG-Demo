import copy
import json
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import trimesh
import vedo
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.drawing.nx_agraph import graphviz_layout
from pog.graph import shape
from pog.graph.edge import Edge
from pog.graph.node import ContainmentState, Node
from pog.graph.params import FRICTION_ANGLE_THRESH, PairedSurface, ROBOT_YML, GRASP_POSE_DATASET
from pog.graph.shapes import Wardrobe, ComplexStorage, Cone, Drawer, Table, Computer
from pog.graph.utils import match
from curobo.geom.types import Mesh
from scipy.spatial.transform import Rotation

import torch
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from curobo.geom.sdf.world import CollisionCheckerType
import yaml
from scipy.spatial.transform import Rotation as R
import re
import numpy as np


class Graph():
    scene: trimesh.Scene

    def __init__(self,
                 scene_name,
                 file_dir=None,
                 file_name=None,
                 fn=None,
                 **kwargs) -> None:
        """class for scene graph and operations on scene graph

        Args:
            scene_name (str): name of scene
            file_path (path, optional): file path of graph. Defaults to None.
            fn (function, optional): function to create graph. Defaults to None.
        """
        self.name = scene_name
        self.graph = nx.DiGraph()
        self.robot = nx.DiGraph()
        self.robot_root = None

        self.ikcfg_initialized = False
        self.robot_cfg = None
        self.tensor_args = None
        self.grasp_pose_data = None
        self.placeholder_nodes = set()

        if file_dir is None and fn is not None:
            self.node_dict, self.edge_dict, self.root = fn(**kwargs)
        elif file_dir is not None and file_name is not None:
            logging.info('Loading scene {} from file {}'.format(
                self.name, os.path.join(file_dir, file_name)))
            self.json_path = os.path.join(file_dir, file_name)
            self.node_dict = {}
            self.edge_dict = {}
            with open(os.path.join(file_dir, file_name)) as json_file:
                data = json.load(json_file)
                self.root = data['root']
                for n in data['nodes']:
                    if n['shape'] == shape.ShapeID.Sphere.value:
                        temp_node = shape.Sphere.from_saved(n)
                    elif n['shape'] == shape.ShapeID.Box.value:
                        temp_node = shape.Box.from_saved(n)
                    elif n['shape'] == shape.ShapeID.Cylinder.value:
                        temp_node = shape.Cylinder.from_saved(n)
                    elif n['shape'] == shape.ShapeID.Cone.value:
                        temp_node = Cone.from_saved(n)
                    elif n['shape'] == shape.ShapeID.Storage.value:
                        temp_node = shape.Storage.from_saved(n)
                    elif n['shape'] == shape.ShapeID.Imported.value:
                        temp_node = shape.Imported.from_saved(n, file_dir)
                    elif n['shape'] == shape.ShapeID.ComplexStorage.value:
                        temp_node = ComplexStorage.from_saved(n)
                    elif n['shape'] == shape.ShapeID.Wardrobe.value:
                        temp_node = Wardrobe.from_saved(n)
                    elif n['shape'] == shape.ShapeID.Drawer.value:
                        temp_node = Drawer.from_saved(n)
                    elif n['shape'] ==shape.ShapeID.Table.value:
                        temp_node = Table.from_saved(n)
                    elif n['shape'] ==shape.ShapeID.Computer.value:
                        temp_node = Computer.from_saved(n)
                    else:
                        raise Exception('Unsupported shape type: {}'.format(
                            shape.ShapeID(n['shape'])))

                    self.node_dict[n['id']] = Node(id=n['id'], shape=temp_node)

                for e in data['edges']:
                    self.edge_dict[e['child']] = Edge(e['parent'], e['child'])
                    for relation in e['relations'].values():
                        try:
                            self.edge_dict[e['child']].add_relation(
                                self.node_dict[e['parent']].affordance[
                                    relation['parent']],
                                self.node_dict[e['child']].affordance[
                                    relation['child']], relation['dof'],
                                relation['pose'])
                        except KeyError:
                            self.edge_dict[e['child']].add_relation(
                                self.node_dict[e['parent']].affordance[
                                    relation['parent']],
                                self.node_dict[e['child']].affordance[
                                    relation['child']])

        else:
            logging.error('Requires an initialization method of graph.')

        for _, node in self.node_dict.items():
            self.graph.add_node(node.id, node=node)

        for _, edge in self.edge_dict.items():
            self.graph.add_edge(edge.parent_id, edge.child_id, edge=edge)
            if edge.parent_id == self.root:
                self.root_aff = edge.relations[
                    shape.AffordanceType.Support]['parent'].name

        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return nx.algorithms.isomorphism.is_isomorphic(self.graph,
                                                       other.graph,
                                                       node_match=match,
                                                       edge_match=match)
    
    def initIKConfig(self):
        if not self.ikcfg_initialized:
            self.robot_cfg = load_yaml(ROBOT_YML)["robot_cfg"]
            self.tensor_args = TensorDeviceType()
            with open(GRASP_POSE_DATASET, 'r', encoding="utf-8") as file:
                self.grasp_pose_data = yaml.safe_load(file)  
            logging.getLogger("curobo").setLevel(logging.WARNING)
            self.ikcfg_initialized = True

    def removeEdge(self, child_id):
        """remove an edge from edge list

        Args:
            child_id (int): node id of child node of removed edge
        """
        del self.edge_dict[child_id]

    def removeNode(self, id, edge=None, ee=None):
        """Remove environment node [id], all its child nodes and their adjacent edges. 
        Add removed environment nodes to robot graph and attach to ee.

        Args:
            id (int): node id
            ee (int): end-effector node in robot graph

        """
        if not self.graph.has_node(id):
            logging.error(
                'GraphBase.removeNode(): Cannot find node {} in environment graph!'
                .format(id))
        if ee is not None and not self.robot.has_node(ee):
            logging.error(
                'GraphBase.removeNode(): Cannot find end effector {} in robot graph!'
                .format(ee))
        if id == self.root:
            logging.error('GraphBase.removeNode(): Cannot remove root node!')

        self.graph.remove_node(id)
        self.robot.add_node(id, node=self.node_dict[id])
        self.__removeNodeHelper(id)

        self.removeEdge(id)

        if ee is not None:
            assert edge is not None
            self.robot.add_edge(ee, id, edge=edge)
        else:
            self.robot_root = id

        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def __removeNodeHelper(self, id):
        for child, edge in self.edge_dict.items():
            if edge.parent_id == id and self.graph.has_node(child):
                self.__removeNodeHelper(child)
                self.robot.add_node(child, node=self.node_dict[child])
                self.robot.add_edge(edge.parent_id,
                                    child,
                                    edge=self.edge_dict[child])
                self.graph.remove_node(child)

    def addExternalNode(self, node: Node, edge: Edge):
        """Add external node to scene graph

        Args:
            node (Node): external node to be added
            edge (Edge): edge between external node and existing node in scene graph
        """
        assert node.id not in self.node_dict.keys()
        assert node.id == edge.child_id
        assert edge.parent_id in self.node_dict.keys()

        self.node_dict[node.id] = node
        self.edge_dict[node.id] = edge
        self.graph.add_node(node.id, node=node)
        self.graph.add_edge(edge.parent_id, edge.child_id, edge=edge)

        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def addNode(self, parent, edge, object=None):
        """Add node to environment graph, remove node from robot graph

        Args:
            parent (int): The node on environment graph that we move the object to
            edge (Edge): The new edge established between parent and object
            object (int, optional): The node on robot graph that we want to move to environment graph. Defaults to None (Move all objects).
        """
        if not self.graph.has_node(parent):
            logging.error(
                'GraphBase.addNode(): Cannot find node {} in environment graph!'
                .format(parent))
        if object is not None and not self.robot.has_node(object):
            logging.error(
                'GraphBase.addNode(): Cannot find node {} in robot graph!'.
                format(object))

        if object is None or object == self.robot_root:
            self.graph = nx.compose(self.graph, self.robot)
            self.robot.clear()
            self.graph.add_edge(parent, self.robot_root, edge=edge)
            self.edge_dict[self.robot_root] = edge
        else:
            self.robot.remove_node(object)
            self.graph.add_node(object, node=self.node_dict[object])
            self.graph.add_edge(parent, object, edge=edge)
            self.__addNodeHelper(id)
            self.removeEdge(object)
            self.edge_dict[self.robot_root] = edge

        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def __addNodeHelper(self, object):
        for child, parent in self.edge_dict.items():
            if parent == object and self.robot.has_node(child):
                self.__addNodeHelper(child)
                self.graph.add_node(child, node=self.node_dict[child])
                self.graph.add_edge(parent, child, edge=self.edge_dict[child])
                self.robot.remove_node(child)

    def toJson(self, file_dir: str = None, file_name = None, init_json: str = None):
        os.makedirs(file_dir, exist_ok=True)
        nodes = []
        edges = []

        try:
            with open(init_json, 'r') as json_file:
                init_data = json.load(json_file)
            for node in init_data['nodes']:
                if node['id'] in self.node_dict.keys():
                    if hasattr(self.node_dict[node['id']], 'state'):
                        node['state'] = 1 \
                            if self.node_dict[node['id']].state == ContainmentState.Opened else 0
                    nodes.append(node)
        except:
            logging.warning(f"Initial JSON file {init_json} not found. Using toJsonLegacy().")
            self.toJsonLegacy(file_dir, file_name)
            return

        for _, edge in self.edge_dict.items():
            temp_edge = {}
            temp_edge['parent'] = edge.parent_id
            temp_edge['child'] = edge.child_id
            temp_edge['relations'] = {}
            for key, value in edge.relations.items():
                temp_relation = {}
                temp_relation['parent'] = getattr(value['parent'], 'name', '')
                temp_relation['child'] = getattr(value['child'], 'name', '')
                try:
                    temp_relation['dof'] = value['dof']
                    temp_relation['pose'] = list(value['pose'])
                except KeyError:
                    pass
                temp_edge['relations'][key.name] = temp_relation
            edges.append(temp_edge)

        with open(os.path.join(file_dir, file_name), 'w') as outfile:
            logging.info('Saving scene {} to file {}'.format(
                self.name, os.path.join(file_dir, file_name)))
            json.dump({
                'nodes': nodes,
                'edges': edges,
                'root': self.root
            }, outfile)


    def toJsonLegacy(self, file_dir: str = None, file_name=None):
        os.makedirs(file_dir, exist_ok=True)
        os.makedirs(file_dir + '/meshes', exist_ok=True)

        nodes = []
        edges = []

        for node_id, node in self.node_dict.items():
            temp_node = {}
            temp_node['id'] = node.id
            temp_node['shape'] = node.shape.shape_type.value
            temp_node['transform'] = node.shape.transform.tolist()
            temp_node['radius'] = getattr(node.shape, 'radius', -1.)
            temp_node['size'] = getattr(node.shape, 'size',
                                        np.array([])).tolist()
            temp_node['height'] = getattr(node.shape, 'height', -1.)
            if node.shape.shape_type == shape.ShapeID.Imported:
                try:
                    temp_node['file_path'] = node.shape.mesh_dir
                    print(node.shape.mesh_dir)
                except:
                    node.shape.shape.export(file_dir + '/meshes/' +
                                            str(node_id) + '.stl')
                    temp_node['file_path'] = file_dir + '/meshes/' + str(
                        node_id) + '.stl'

            nodes.append(temp_node)

        for _, edge in self.edge_dict.items():
            temp_edge = {}
            temp_edge['parent'] = edge.parent_id
            temp_edge['child'] = edge.child_id
            temp_edge['relations'] = {}
            for key, value in edge.relations.items():
                temp_relation = {}
                temp_relation['parent'] = getattr(value['parent'], 'name', '')
                temp_relation['child'] = getattr(value['child'], 'name', '')
                try:
                    temp_relation['dof'] = value['dof']
                    temp_relation['pose'] = list(value['pose'])
                except KeyError:
                    pass
                temp_edge['relations'][key.name] = temp_relation
            edges.append(temp_edge)

        with open(os.path.join(file_dir, file_name), 'w') as outfile:
            logging.info('Saving scene {} to file {}'.format(
                self.name, os.path.join(file_dir, file_name)))
            json.dump({
                'nodes': nodes,
                'edges': edges,
                'root': self.root
            }, outfile)

    def getPose(self, edge_id=None):
        """Get pose of current graph

        Args:
            edge_id (list, optional): A list of edges (child nodes). Defaults to None.

        Returns:
            pose_dict (dict (child, pose)): child node id and its pose
        """
        pose_dict = {}
        if edge_id is None:
            for _, edge in self.edge_dict.items():
                if self.graph.has_node(edge.parent_id) and self.graph.has_node(
                        edge.child_id):
                    pose_dict[edge.child_id] = edge.relations[
                        shape.AffordanceType.Support]
        else:
            for edge in edge_id:
                if self.graph.has_node(self.edge_dict[edge].parent_id
                                       ) and self.graph.has_node(
                                           self.edge_dict[edge].child_id):
                    pose_dict[self.edge_dict[edge].child_id] = self.edge_dict[
                        edge].relations[shape.AffordanceType.Support]

        return pose_dict

    def setPose(self, pose):
        """Set pose of scene graph

        Args:
            pose (dict (child, pose)): Child node id and its pose
        """
        for child_id, relation in pose.items():
            if self.graph.has_node(
                    relation['parent'].node_id) and self.graph.has_node(
                        relation['child'].node_id):
                self.edge_dict[child_id].add_relation(relation['parent'],
                                                      relation['child'],
                                                      relation['dof'],
                                                      relation['pose'])
                self.graph.add_edge(relation['parent'].node_id,
                                    relation['child'].node_id,
                                    edge=self.edge_dict[child_id])
        self.updateCoM()
        self.computeGlobalTF()

    def trackDepth(self):
        """Find nodes at each depth and store it in self.depth_dict
        """
        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        node_depth = nx.shortest_path_length(self.graph, self.root)
        self.depth_dict = {}
        for depth in range(0, max_depth):
            temp_depth_list = []
            for key, value in node_depth.items():
                if value == depth:
                    temp_depth_list.append(key)
            self.depth_dict[depth] = temp_depth_list

    def updateAccessibility(self,
                            robot_at_node_id=None):  # For pick and place only
        # TODO: the robot is not at root. Only at root for now
        # if robot_at_node_id is None or robot_at_node_id == self.root:
        #     robot_at_node_id = self.root
        #     self.node_dict[self.root].accessible = True

        max_depth = len(nx.algorithms.dag_longest_path(self.graph))

        for node in self.node_dict.values():
            node.accessible = True

        checked_nodes = []
        for depth in range(1, max_depth):
            node_list_current_depth = self.depth_dict[depth]
            for current_depth_node in node_list_current_depth:
                if self.edge_dict[current_depth_node].containment and \
                self.node_dict[self.edge_dict[current_depth_node].parent_id].state == ContainmentState.Closed and \
                    current_depth_node not in checked_nodes:
                    sub_tree = dfs_tree(self.graph, current_depth_node)
                    for sub_tree_node in sub_tree.nodes:
                        self.node_dict[sub_tree_node].accessible = False
                        checked_nodes.append(sub_tree_node)

    def updateCoM(self):
        """
            Recursively compute center of mass of current scene and store it in edges. 
            CoM stored in each edge is the CoM for all its children.
        """
        self.trackDepth()
        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        for i in reversed(range(1, max_depth)):
            node_dict = self.depth_dict[i]
            for node_id in node_dict:
                total_mass = self.node_dict[node_id].shape.mass
                self.edge_dict[node_id].relations[
                    shape.AffordanceType.Support]['mass'] = total_mass
                self.edge_dict[node_id].relations[shape.AffordanceType.Support]['com'] = \
                    np.dot(self.edge_dict[node_id].parent_to_child_tf, np.concatenate((self.node_dict[node_id].shape.com, [1])))[0:3] * total_mass
                for succ in self.graph.successors(node_id):
                    self.edge_dict[node_id].relations[shape.AffordanceType.Support]['mass'] += \
                        self.edge_dict[succ].relations[shape.AffordanceType.Support]['mass']
                    total_mass += self.edge_dict[succ].relations[
                        shape.AffordanceType.Support]['mass']
                    self.edge_dict[node_id].relations[shape.AffordanceType.Support]['com'] += \
                        np.dot(self.edge_dict[node_id].parent_to_child_tf,
                        np.concatenate((self.edge_dict[succ].relations[shape.AffordanceType.Support]['com'], [1])))[0:3] * \
                            self.edge_dict[succ].relations[shape.AffordanceType.Support]['mass']
                self.edge_dict[node_id].relations[
                    shape.AffordanceType.Support]['com'] /= total_mass

    def show(self):
        """Show scene graph and robot graph
        """
        plt.figure()
        node_labels = {}
        for key in self.node_dict:
            if self.graph.has_node(key):
                node_labels[key] = str(self.node_dict[key])
        pos = graphviz_layout(self.graph, prog="dot")
        nx.draw_networkx_nodes(self.graph,
                               pos,
                               cmap=plt.get_cmap('jet'),
                               node_size=1500)
        nx.draw_networkx_labels(self.graph,
                                pos,
                                labels=node_labels,
                                font_weight="bold")
        nx.draw_networkx_edges(self.graph,
                               pos,
                               edgelist=self.graph.edges,
                               edge_color='r',
                               arrows=True,
                               width=2,
                               arrowsize=50)
        plt.title('Environment Model')

        plt.figure()
        node_labels = {}
        for key in self.node_dict:
            if self.robot.has_node(key):
                node_labels[key] = str(self.node_dict[key])
        pos = graphviz_layout(self.robot, prog="dot")
        nx.draw_networkx_nodes(self.robot,
                               pos,
                               cmap=plt.get_cmap('jet'),
                               node_size=1500)
        nx.draw_networkx_labels(self.graph,
                                pos,
                                labels=node_labels,
                                font_weight="bold")
        nx.draw_networkx_edges(self.robot,
                               pos,
                               edgelist=self.robot.edges,
                               edge_color='r',
                               arrows=True,
                               width=2,
                               arrowsize=50)
        plt.title('Robot Model')

        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
        plt.close()


    def genNodeShape(self, object_id: int) -> trimesh.Trimesh:
        
        node = self.node_dict[object_id]
        try: node_shape = node.shape.shape
        except AttributeError as e: 
            raise Exception(f"Object_{object_id} has no shape: {e}")
        
        if node.shape.object_type == shape.ShapeType.ARTIC and node.state: 
            if node.state == ContainmentState.Opened and hasattr(node.shape, 'open_shape'):
                node_shape = node.shape.open_shape
            elif node.state == ContainmentState.Closed and hasattr(node.shape, 'close_shape'):
                node_shape = node.shape.close_shape

        return node_shape.copy()


    def create_scene(self):
        """create Trimesh.Scene for visualization
        """
        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        node_depth = nx.shortest_path_length(self.graph, self.root)
        geom = self.genNodeShape(self.root)
        self.scene = trimesh.Scene()
        self.scene.add_geometry(geom, node_name=self.root)
        for i in range(1, max_depth):
            for key, value in node_depth.items():
                if value == i:
                    node_geom = self.genNodeShape(key)
                    self.scene.add_geometry(
                        node_geom,
                        node_name=key,
                        parent_node_name=self.edge_dict[key].parent_id,
                        transform=self.edge_dict[key].parent_to_child_tf)
    

                    

    def create_collision_scene(self):
        """create Trimesh.Scene for visualization
        """
        self.collision_scene = trimesh.Scene()

        for key, _ in self.node_dict.items():
            if key in self.graph.nodes():
                try:
                    self.collision_scene.add_geometry(
                        geometry=self.genNodeShape(key),
                        node_name=str(key),
                        transform=self.global_transform[key])
                except:
                    logging.info(f"{key}, {self.global_transform.keys()}")

    @staticmethod
    def transform_matrix_to_list(matrix: np.ndarray) -> list:
        """
        Convert a 4x4 transformation matrix to a list [x, y, z, qw, qx, qy, qz].
        """
        assert matrix.shape == (4, 4), "Input must be a 4x4 transformation matrix"
        x, y, z = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
        return [x, y, z, quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

    def genMeshList(self, exclude_object_ids = None):

        """  Generate curobo-type Mesh list for obstacle representation. """

        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        node_depth = nx.shortest_path_length(self.graph, self.root)

        mesh_list = []
        self.computeGlobalTF()

        for depth in range(1, max_depth):
            for node_id in node_depth:
                if node_id not in (exclude_object_ids or set()):
                    if node_depth[node_id] == depth:
                        node_mesh = self.genNodeShape(node_id).copy()
                        # transform_pose = [0, 0, 0, 1, 0, 0, 0]
                        transform_pose = self.transform_matrix_to_list(self.global_transform[node_id])

                        mesh = Mesh(
                            name=f"object_{node_id}",
                            pose=transform_pose,
                            vertices=node_mesh.vertices.tolist(),
                            faces=node_mesh.faces.tolist()
                        )
                        mesh_list.append(mesh)
        return mesh_list

    def genWorldCfg(self, exclude_object_ids = None):
        world_cfg = WorldConfig(mesh=self.genMeshList(exclude_object_ids))
        return world_cfg

    def genMesh(self, outfile='out.stl'):
        """create mesh of scene graph and save it to a directory

        Args:
            outfile (str, optional): file path. Defaults to 'out.stl'.
        """
        self.create_scene()
        self.scene.export(outfile)

    def computeGlobalTF(self):
        """Compute transformations from root to all nodes in scene graph 
        """
        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        node_depth = nx.shortest_path_length(self.graph, self.root)
        self.global_transform = {}
        self.global_transform[self.root] = np.identity(4)
        for i in range(1, max_depth):
            for key, value in node_depth.items():
                if value == i:
                    tf = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                                   (0, 0, 0, 1)))
                    tf = self.__genGlobalTFHelper(key, value, tf)
                    try:
                        self.global_transform[key] = tf
                    except KeyError:
                        logging.error(
                            'Cannot find edge of child ID {} in edge list.'.
                            format(key))

    def __genGlobalTFHelper(self, key, value, tf):

        if value == 0:
            return tf
        elif value > 0:
            value -= 1
            tf = np.dot(self.edge_dict[key].parent_to_child_tf, tf)
            return self.__genGlobalTFHelper(self.edge_dict[key].parent_id,
                                            value, tf)

    def getSubGraph(self, node_id=None, depth=None):
        """Get subgraph given a node in current scene graph. Return a deepcopy of self if node_id is None

        Args:
            node_id (int): node id
            depth (int, optional): The depth of subgraph. Defaults to None.

        Returns:
            graph (Graph): The subgraph of scene graph with root at nood_id
        """
        if node_id is None:
            node_id = self.root
        else:
            assert self.graph.has_node(node_id)
        sg = dfs_tree(self.graph, node_id, depth_limit=depth)

        def fn():
            node_dict = {}
            edge_dict = {}

            for item in sg.nodes:
                node_dict[item] = copy.deepcopy(self.node_dict[item])

            for item in sg.edges:
                edge_dict[item[1]] = copy.deepcopy(self.edge_dict[item[1]])

            root_id = node_id
            return node_dict, edge_dict, root_id

        # sub_graph = Graph('Subgraph of {} at node {}.'.format(self.name, node_id), fn = fn)
        sub_graph = Graph('subgraph', fn=fn)
        
        sub_collision_scene = self.collision_scene.copy()
        for key, _ in self.node_dict.items():
            if key not in sub_graph.graph.nodes():
                sub_collision_scene.delete_geometry(str(key))
        sub_graph.collision_scene = sub_collision_scene
        sub_graph.collision_manager, _ = trimesh.collision.scene_to_collision(sub_collision_scene)

        return sub_graph

    def copy(self):

        def fn():
            node_dict = {}
            edge_dict = {}

            for item in self.graph.nodes:
                node_dict[item] = copy.deepcopy(self.node_dict[item])

            for item in self.graph.edges:
                if item != self.root:
                    edge_dict[item[1]] = copy.deepcopy(self.edge_dict[item[1]])

            root_id = self.root
            return node_dict, edge_dict, root_id

        graph_copy = Graph('graph copy', fn=fn)

        # print(nx.algorithms.tree.recognition.is_tree(self.graph), nx.algorithms.tree.recognition.is_tree(graph_copy.graph))
        # print(self.edge_dict.keys(), graph_copy.edge_dict.keys())
        # print(self.node_dict.keys(), graph_copy.node_dict.keys())
        # print(self.graph.edges, graph_copy.graph.edges)
        # print(self.edge_dict.values())

        graph_copy.root_aff = self.root_aff
        graph_copy.ikcfg_initialized = self.ikcfg_initialized
        graph_copy.robot_cfg = self.robot_cfg
        graph_copy.tensor_args = self.tensor_args
        graph_copy.grasp_pose_data = self.grasp_pose_data

        graph_copy.placeholder_nodes = self.placeholder_nodes.copy()    

        if self.robot_root is not None:
            graph_copy.robot = self.robot.copy()
            graph_copy.robot_root = self.robot_root

            for item in self.robot.nodes:
                graph_copy.node_dict[item] = copy.deepcopy(
                    self.node_dict[item])

            for item in self.robot.edges:
                if item != self.robot_root:
                    graph_copy.edge_dict[item[1]] = copy.deepcopy(
                        self.edge_dict[item[1]])

        return graph_copy

    def createCollisionManagerLegacy(self):
        """Create Trimesh.collision.CollisionManager for current scene
        """
        self.collision_manager = trimesh.collision.CollisionManager()
        for key, _ in self.node_dict.items():
            if key in self.graph.nodes():
                try:
                    self.collision_manager.add_object(
                        name=str(key),
                        mesh=self.genNodeShape(key),
                        transform=self.global_transform[key])
                except:
                    print(key, self.global_transform.keys())

    def createCollisionManager(self):
        self.create_collision_scene()
        self.collision_manager, _ = trimesh.collision.scene_to_collision(self.collision_scene)

    def checkStability(self):
        """Check if self is stable

        Returns:
            (bool): True if self is stable
        """
        stable = True
        unstable_node = []
        self.computeGlobalTF()
        vertical_dir = self.node_dict[self.root].affordance[
            self.root_aff].get_axes()
        for node_id in self.edge_dict.keys():
            if node_id not in self.graph.nodes: continue
            parent_id = self.edge_dict[node_id].parent_id
            parent_aff_name = self.edge_dict[node_id].relations[
                shape.AffordanceType.Support]['parent'].name
            tf = self.global_transform[parent_id] @ self.node_dict[
                parent_id].affordance[parent_aff_name].transform
            uv1 = tf[0:3, 2] / np.linalg.norm(tf[0:3, 2])
            uv2 = vertical_dir / np.linalg.norm(vertical_dir)
            angle = np.arccos(np.dot(uv1, uv2))
            if angle > FRICTION_ANGLE_THRESH:
                stable = False
                unstable_node.append(node_id)
        return (stable, unstable_node)

    @staticmethod
    def quaternion_multiply(q1, q2):

        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)   
    
    def CreateIKSolver(self, world_cfg = None):

        self.initIKConfig()

        if world_cfg is None:
            world_cfg = WorldConfig(mesh=self.genMeshList())

        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            world_cfg,
            rotation_threshold=0.5,
            position_threshold=0.1,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=False,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"mesh": 10},
            # use_fixed_samples=True,
        )
        return IKSolver(ik_config)

    def genGraspPoses(self, object_id, object_pose = None):
        """
        Generate grasp poses (relative to root frame) for object with id object_id

        Args:
            object_id (int): object id
            object_pose (list, optional): object pose [x, y, z, qw, qx, qy, qz]. Defaults to None.

        Returns: 
            positions, quaternion  (torch.tensor)
        """

        if f"object_{object_id}" in self.grasp_pose_data:
            grasp_poses = self.grasp_pose_data[f"object_{object_id}"]
        else:
            raise KeyError(f"Missing grasp pose data for object_{object_id}")

        if object_pose is None:
            self.computeGlobalTF()
            object_pose = self.transform_matrix_to_list(self.global_transform[object_id])

        positions = [torch.tensor(pose['position'], dtype=torch.float32, device="cuda:0")
                    for pose in grasp_poses]
        orientations = [torch.tensor(pose['orientation'], dtype=torch.float32, device="cuda:0")
                        for pose in grasp_poses]

        pose_positions = torch.stack(positions)
        pose_orientations = torch.stack(orientations)

        T_position = torch.tensor(object_pose[:3], dtype=torch.float32, device="cuda:0")
        T_quaternion = torch.tensor(object_pose[3:], dtype=torch.float32, device="cuda:0")

        T_rotation = R.from_quat(np.array(object_pose[3:]), scalar_first=True).as_matrix()
        T_rotation = torch.tensor(T_rotation, dtype=torch.float32, device="cuda:0")

        rotated_positions = pose_positions @ T_rotation.T
        new_positions = rotated_positions + T_position
        new_quaternions = self.quaternion_multiply(T_quaternion.unsqueeze(0), pose_orientations)   

        return new_positions, new_quaternions  

    def checkIK(self, object_id, ik_solver=None, object_pose=None, get_succ_rate=False):
        """
        Check if the grasp poses for a given object are collision-aware feasible.

        Args:
            object_id (int): The ID of the object to check.
            ik_solver (IKSolver, optional): The IK solver instance. Defaults to None.
            object_pose (list, optional): The pose of the object [x, y, z, qw, qx, qy, qz]. Defaults to None.
            get_succ_rate (bool, optional): Whether to return the success rate of IK solutions. Defaults to False.

        Returns:
            bool or float: If `get_succ_rate` is False, returns True if at least one IK solution exists. 
                           If `get_succ_rate` is True, returns the success rate as a float.
        """
        if f"object_{object_id}" not in self.grasp_pose_data:
            logging.info(f"Grasp poses for object_{object_id} are not defined. Defaulting IK check to True.")
            return True

        if ik_solver is None:
            ik_solver = self.CreateIKSolver()

        positions, quaternions = self.genGraspPoses(object_id, object_pose)

        if get_succ_rate:
            goal = Pose(positions, quaternions)
            result = ik_solver.solve_batch(goal)
            torch.cuda.synchronize()
            success_rate = torch.count_nonzero(result.success).item() / len(goal)
            return success_rate
        else:
            goal = Pose(positions.unsqueeze(0), quaternions.unsqueeze(0))
            result = ik_solver.solve_goalset(goal)
            torch.cuda.synchronize()
            return torch.any(result.success).item()

