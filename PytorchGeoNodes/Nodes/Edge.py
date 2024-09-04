import torch


class Socket(torch.nn.Module):
    def __init__(self, bpy_socket):
        super().__init__()

        self.name = bpy_socket.name
        self.type = bpy_socket.type

        self.identifier = bpy_socket.identifier


class OutEdge(torch.nn.Module):
    def __init__(self, to_node, bpy_link):
        super().__init__()

        self.to_node = to_node

        self.from_node_name = bpy_link.from_node.name
        self.to_node_name = bpy_link.to_node.name
        self.from_socket = Socket(bpy_link.from_socket)
        self.to_socket = Socket(bpy_link.to_socket)

class InEdge(torch.nn.Module):
    def __init__(self, from_node, bpy_link):
        super().__init__()

        self.from_node = from_node

        self.from_node_name = bpy_link.from_node.name
        self.to_node_name = bpy_link.to_node.name
        self.from_socket = Socket(bpy_link.from_socket)
        self.to_socket = Socket(bpy_link.to_socket)
