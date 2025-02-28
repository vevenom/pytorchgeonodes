import torch.nn
import numpy as np
from copy import copy
from multiprocessing import current_process

import yaml

from utils import DictAsMember

if current_process().name == 'MainProcess':
    import bpy
    import open3d as o3d

import random
import dill


from pytorch3d.transforms import Transform3d
from pytorch3d.structures import Meshes

from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram
from PytorchGeoNodes.Nodes.node_types import NodeTypes
from PytorchGeoNodes.Nodes.NodeInputsDict import NodeInputsDict
from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.NodeCombXYZ import NodeCombXYZ
from PytorchGeoNodes.Nodes.NodeMath import NodeMath
from PytorchGeoNodes.Nodes.NodeMeshPrimitiveLine import NodeMeshPrimitiveLine
from PytorchGeoNodes.Nodes.NodeMeshPrimitiveCube import NodeMeshPrimitiveCube
from PytorchGeoNodes.Nodes.NodeMeshPrimitiveCylinder import NodePrimitiveCylinder
from PytorchGeoNodes.Nodes.NodeInstanceOnPoints import NodeInstanceOnPoints
from PytorchGeoNodes.Nodes.NodeGroupOutput import NodeGroupOutput
from PytorchGeoNodes.Nodes.NodeGroupInput import NodeGroupInput
from PytorchGeoNodes.Nodes.NodeGroupInputNested import NodeGroupInputNested
from PytorchGeoNodes.Nodes.NodeRealizeInstances import NodeRealizeInstances
from PytorchGeoNodes.Nodes.NodeTransformGeometry import NodeTransformGeometry
from PytorchGeoNodes.Nodes.NodeVectMath import NodeVectMath
from PytorchGeoNodes.Nodes.NodeJoinGeometry import NodeJoinGeometry
from PytorchGeoNodes.Nodes.NodeSwitch import NodeSwitch
from PytorchGeoNodes.Nodes.NodeGroup import NodeGroup
from PytorchGeoNodes.Nodes.vis_utils import *
from PytorchGeoNodes.Nodes.Edge import InEdge, OutEdge

class Counters():
    def __init__(self):
        self.total_number_base_primitives = 0

    def increment_total_number_base_primitives(self):
        self.total_number_base_primitives += 1

class GeometryNodes(torch.nn.Module):
    def __init__(self, shape_program: BlenderShapeProgram, node_tree=None, name='GeometryNodes', counters=Counters(),
                 config_path='configs/geometry_nodes_config.yaml'):
        super().__init__()
        self.name = name

        self._modifiers_dict = shape_program._modifiers_dict

        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = DictAsMember(self.config)

        if node_tree is None:
            node_tree = bpy.data.node_groups['Geometry Nodes']

        self.is_group_node = False
        if node_tree != bpy.data.node_groups['Geometry Nodes']:
            self.is_group_node = True

        self.counters = counters
        self.nodes, self.nodes_compute_order_list, self.trainable_params  = (
            self._generate_graph_(shape_program, node_tree))

        self.nodes = nn.ModuleList(self.nodes)


        # Default transform
        transform_py3d2blender = np.eye(4)
        transform_py3d2blender[0, 0] = 1
        transform_py3d2blender[1, 1] = 0
        transform_py3d2blender[1, 2] = -1
        transform_py3d2blender[2, 1] = 1
        transform_py3d2blender[2, 2] = 0
        transform_py3d2blender = np.linalg.inv(transform_py3d2blender)
        transform_py3d2blender = torch.tensor(transform_py3d2blender, dtype=torch.float32)
        self.transform_py3d2blender = Transform3d(matrix=transform_py3d2blender)

    def to(self, device):
        super().to(device)
        for node in self.nodes:
            node.to(device)
        self.transform_py3d2blender = self.transform_py3d2blender.to(device)

    def get_device(self):
        return self.transform_py3d2blender.device

    def draw_graph(self):
        import networkx
        import matplotlib.pyplot as plt

        G = networkx.DiGraph()
        for node in self.nodes:
            G.add_node(node.name)
            for edge in node.out_edges:
                edge = edge[1]
                G.add_edge(edge.from_node.name, edge.to_node.name)

        networkx.draw(G, with_labels=True)
        plt.show()

    def forward_draw_graph(self, inputs_dict):
        import networkx
        import matplotlib.pyplot as plt

        inputs_dict, _ = self.forward(inputs_dict, transform2blender_coords=False)

        G = networkx.DiGraph()
        labels = {}
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        node_type_color = {}
        node_names_colors = {}
        for node in self.nodes:

            node_type = node.type
            # if node_type is already in node_type_color.keys() assign color otherwise generate new color
            # node_type_color[node_type] = (
            #     node_type_color)[node_type] if node_type in node_type_color.keys() else random.choice(colors)
            node_names_colors[node.name] = GeometryNodesVis.node_type_colors[node_type]

            # add colored node to graph and color it
            G.add_node(node.name)

            for edge in node.out_edges:
                edge = edge[1]

                G.add_edge(edge.from_node.name, edge.to_node.name)

                value = inputs_dict[node.name][NodeStrings.OUT_str + edge.from_socket.identifier]
                if torch.is_tensor(value):
                    value = torch.round(value * 100.0) / 100.0
                    value = str(value)
                    value = value.replace('tensor', '')
                    # show only 3 decimals
                    value = value[:value.find('.') + 4]
                    labels[(edge.from_node.name, edge.to_node.name)] = value

        # add colors
        colors_data = []
        for i in range(len(G.nodes)):
            colors_data.append(node_names_colors[list(G.nodes)[i]])

        pos = networkx.spring_layout(G, k=50/np.sqrt(G.order()))
        # pos = networkx.planar_layout(G, scale=5)
        # pos = networkx.multipartite_layout(G)

        networkx.draw_networkx_nodes(G, pos, node_color=colors_data)
        networkx.draw_networkx_edges(G, pos, edge_color='tab:red')
        networkx.draw_networkx_labels(G, pos)
        networkx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        # networkx.draw(G, with_labels=True)
        plt.show()
        assert False

    def _generate_graph_(self, blender_shape_program: BlenderShapeProgram, node_tree, check_pickable=False):

        # node_tree is not really a tree, it is an acyclic graph

        # Create nodes
        nodes = torch.nn.Sequential()
        nodes_dict = {}
        trainable_params = nn.ModuleList()
        for bpy_node in node_tree.nodes:
            torch_node = self.create_node(bpy_node, blender_shape_program)
            if torch_node is None:
                continue
            nodes.append(torch_node)
            nodes_dict[torch_node.name] = torch_node

        # Create edges
        for bpy_link in node_tree.links:
            from_node = bpy_link.from_node
            to_node = bpy_link.to_node
            from_socket = bpy_link.from_socket
            to_socket = bpy_link.to_socket

            from_node = nodes_dict[from_node.name]
            to_node = nodes_dict[to_node.name]

            out_edge = OutEdge(to_node, bpy_link)
            in_edge = InEdge(from_node, bpy_link)

            from_node.add_out_edge(out_edge)
            to_node.add_in_edge(in_edge)

        nodes_ordered, nodes_compute_order_list = GeometryNodes.sort_nodes(nodes)

        if check_pickable:
            for node in nodes_ordered:
                assert dill.pickles(node), 'Node %s is not pickable' % node.name
            print("All nodes are pickable for graph %s ." % self.name)

        return nodes_ordered, nodes_compute_order_list, trainable_params

    @staticmethod
    def sort_nodes(nodes):
        """
        Topological sort nodes such that values can be computed using efficient order
        based on dependencies

        Args:
            nodes:

        Returns:
            nodes_ordered: Ordered nodes
            nodes_compute_order_dict: Dictionary containing nodes in each level
        """

        node_indegrees_rem = {}
        nodes_ordered = torch.nn.Sequential()
        nodes_to_visit = []

        nodes_compute_order_dict = {}
        nodes_compute_order_dict[0] = []
        for node in nodes:
            node_indegrees_rem[node.name] = len(node.in_edges)
            if len(node.in_edges) == 0:
                nodes_to_visit.append(node)
                nodes_compute_order_dict[0].append(node)

        next_compute_level = 1
        # nodes_compute_order_dict[1] = []
        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop()
            if isinstance(node, int):
                next_compute_level = node
                continue
            nodes_ordered.append(node)
            for out_edge in node.out_edges:
                node_indegrees_rem[out_edge.to_node.name] -= 1
                if node_indegrees_rem[out_edge.to_node.name] == 0:
                    nodes_to_visit.append(next_compute_level + 1)
                    nodes_to_visit.append(out_edge.to_node)

                    if next_compute_level not in nodes_compute_order_dict.keys():
                        nodes_compute_order_dict[next_compute_level] = []

                    # check if out_node is dependent on any node in next_compute_level and push it to next level if so
                    out_in_nodes = [edge.from_node for edge in out_edge.to_node.in_edges]
                    level_keys = list(nodes_compute_order_dict.keys())
                    # level_keys = [key for key in level_keys if key >= next_compute_level]
                    level_keys.sort(reverse=True)
                    found_dep_n = False
                    for level_key in level_keys:
                        for out_in_n in out_in_nodes:
                            if out_in_n in nodes_compute_order_dict[level_key]:
                                if level_key + 1 not in nodes_compute_order_dict.keys():
                                    nodes_compute_order_dict[level_key + 1] = []
                                nodes_compute_order_dict[level_key + 1].append(out_edge.to_node)
                                found_dep_n = True
                                break
                        if found_dep_n:
                            break
                    if found_dep_n:
                        continue

                    nodes_compute_order_dict[next_compute_level].append(out_edge.to_node)

        nodes_compute_order_list = [None] * len(nodes_compute_order_dict.keys())
        for key in nodes_compute_order_dict.keys():
            nodes_compute_order_list[key] = nodes_compute_order_dict[key]

        return nodes_ordered, nodes_compute_order_list

    def create_node(self, bpy_node, blender_shape_program: BlenderShapeProgram):

        new_node = None  # type: Node
        if bpy_node.type == NodeTypes.MATH_str:
            new_node = NodeMath(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.COMBXYZ_str:
            new_node = NodeCombXYZ(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.GROUP_INPUT_str:
            if self.is_group_node:
                new_node = NodeGroupInputNested(bpy_node, config=self.config)
            else:
                new_node = NodeGroupInput(bpy_node, blender_shape_program.get_params_dict(), config=self.config)
        elif bpy_node.type == NodeTypes.GROUP_OUTPUT_str:
            new_node = NodeGroupOutput(bpy_node, config=self.config)
        # elif bpy_node.type == POINTS_str:
        #     new_node = NodePoints(bpy_node)
        elif bpy_node.type == NodeTypes.MESH_PRIMITIVE_LINE_str:
            new_node = NodeMeshPrimitiveLine(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.MESH_PRIMITIVE_CUBE_str:
            new_node = NodeMeshPrimitiveCube(self.counters.total_number_base_primitives, bpy_node, config=self.config)
            self.counters.increment_total_number_base_primitives()
        elif bpy_node.type == NodeTypes.MESH_PRIMITIVE_CYLINDER_str:
            new_node = NodePrimitiveCylinder(self.counters.total_number_base_primitives, bpy_node, config=self.config)
            self.counters.increment_total_number_base_primitives()
        elif bpy_node.type == NodeTypes.INSTANCE_ON_POINTS_str:
            new_node = NodeInstanceOnPoints(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.REALIZE_INSTANCES_str:
            new_node = NodeRealizeInstances(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.TRANSFORM_GEOMETRY_str:
            new_node = NodeTransformGeometry(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.VECT_MATH_str:
            new_node = NodeVectMath(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.GROUP_str:
            # create a new graph for the group
            new_graph = GeometryNodes(blender_shape_program, node_tree=bpy_node.node_tree, name=bpy_node.name,
                                      counters=self.counters, config_path=self.config_path)
            new_node = NodeGroup(bpy_node, new_graph, config=self.config)
        elif bpy_node.type == NodeTypes.JOIN_GEOMETRY_str:
            new_node = NodeJoinGeometry(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.SWITCH_str:
            new_node = NodeSwitch(bpy_node, config=self.config)
        elif bpy_node.type == NodeTypes.FRAME_str:
            # Frames seem to be used only for better visualization in Blender to seperate different blocks of nodes
            # They are not used in the actual computation
            return None
        else:
            raise Exception(f'Node type {bpy_node.type} is not supported yet.')

        return new_node

    def forward(self, inputs_dict, transform2blender_coords=False, parallel_compute=False):
        """
        Forward pass through the graph.

        :param transform2blender_coords:
        :param parallel_compute:
        :param inputs_dict:
        :return: Returns a list for every output node, where each element is list of meshes for
            corresponding output nodes in the graph. (number of output nodes, number of meshes, batch size)
            (We currently do not support batch size > 1)
        """

        assert not parallel_compute, 'Parallel compute not supported yet.'

        inputs_dict = {
            'GeoNodes Name': self.name,
            NodeTypes.Geometry_Nodes_Inputs_str: copy(inputs_dict),
            NodeTypes.Geometry_Nodes_Outputs_str: {}
        }
        inputs_dict = NodeInputsDict(inputs_dict, self.config)

        if not self.is_group_node:
            for key in list(inputs_dict[NodeTypes.Geometry_Nodes_Inputs_str].keys()):
                inputs_dict[NodeTypes.Geometry_Nodes_Inputs_str][self._modifiers_dict[key]] = (
                    inputs_dict)[NodeTypes.Geometry_Nodes_Inputs_str][key]

        outputs = []
        for node in self.nodes:
            # # calculate time needed for geometry nodes
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            #
            # start.record()
            # out = node.forward(inputs_dict)
            #
            # end.record()
            #
            # torch.cuda.synchronize()
            # print('Time needed for node {} nodes: {}'.format(node.name, start.elapsed_time(end)))

            out = node.forward(inputs_dict)
            if out is not None:
                outputs.append(out)

        if transform2blender_coords and not self.is_group_node:
            for output in outputs:
                for out in output:
                    for o_i, o in enumerate(out):
                        verts = o.verts
                        verts = self.transform_py3d2blender.transform_points(verts)

                        assert verts.shape[0] == 1, 'Batch size > 1 not supported yet.'
                        # mesh = Meshes(verts=[verts[0]], faces=[o.faces[0]])
                        # out[o_i] = mesh

                        o.verts = verts

        elif not self.is_group_node:
            for output in outputs:
                for out in output:
                    for o_i, o in enumerate(out):
                        verts = o.verts

                        # assert verts.shape[0] == 1, 'Batch size > 1 not supported yet.'
                        # mesh = Meshes(verts=[verts[0]], faces=[o.faces[0]])
                        # out[o_i] = mesh

                        o.verts = verts

        return inputs_dict, outputs

    def outputs_to_o3d(self, outputs, b_index=0, compute_normals=True):
        '''
        Convert outputs from forward pass to Open3D objects.

        :param outputs:
        :return:
        '''

        outputs_o3d = []
        for output in outputs:
            output_o3d = []
            output = output[b_index]
            for mesh in output:
                verts_np = mesh.verts.detach().cpu().numpy()[0]
                faces_np = mesh.faces.detach().cpu().numpy()[0]

                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
                if compute_normals:
                    o3d_mesh.compute_vertex_normals()

                output_o3d.append(o3d_mesh)
            outputs_o3d.append(output_o3d)
        return outputs_o3d

    def o3d_visualize_colorize_base_primitives(self, outputs, b_index, return_geometry=False, show=False):
        from PytorchGeoNodes.utils import colormap

        outputs_o3d = []
        for output in outputs:
            output_o3d = []
            output = output[b_index]
            for mesh in output:
                verts_np = mesh.verts.detach().cpu().numpy()[0]
                faces_np = mesh.faces.detach().cpu().numpy()[0]

                base_prim_ids_np = mesh.verts_base_primitive_ids.detach().cpu().numpy().astype(np.int32)[0]
                colors_np = colormap[base_prim_ids_np + 1]

                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_np)

                output_o3d.append(o3d_mesh)
            outputs_o3d.extend(output_o3d)

        if show:
            o3d.visualization.draw_geometries(outputs_o3d)
        if return_geometry:
            return outputs_o3d

    def o3d_visualize_colorize_individual_primitives(self, outputs, b_index, return_geometry=False, show=False):
        from PytorchGeoNodes.utils import colormap

        outputs_o3d = []
        for output in outputs:
            output_o3d = []
            output = output[b_index]
            for mesh in output:
                verts_np = mesh.verts.detach().cpu().numpy()[0]
                faces_np = mesh.faces.detach().cpu().numpy()[0]

                ind_prim_ids_np = mesh.verts_individual_primitive_ids.detach().cpu().numpy().astype(np.int32)[0]
                colors_np = colormap[ind_prim_ids_np + 1]

                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_np)

                output_o3d.append(o3d_mesh)
            outputs_o3d.extend(output_o3d)

        if show:
            o3d.visualization.draw_geometries(outputs_o3d)
        if return_geometry:
            return outputs_o3d
