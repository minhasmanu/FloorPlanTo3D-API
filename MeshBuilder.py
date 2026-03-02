
from typing import Any
from matplotlib.dates import num2date
import numpy as np
from gltflib import (GLTF, Accessor, AccessorType, Asset, Attributes, Buffer,
                     BufferView, ComponentType, FileResource, GLTFModel, Mesh,
                     Node, Primitive, PrimitiveMode, Scene)
from gltflib.models import Material, PBRMetallicRoughness

def create(list: list, resource: "Any"):
    id = len(list)
    list.append(resource)
    return id

class MeshBuilder:
    def __init__(self) -> None:
        self.gltf_nodes: "list[Node]" = []
        self.gltf_buffers: "list[Buffer]" = []
        self.gltf_resources: "list[Any]" = []
        self.gltf_buffer_views: "list[BufferView]" = []
        self.gltf_accessors: "list[Accessor]" = []
        self.gltf_meshes: "list[Mesh]" = []
        self.gltf_materials: "list[Material]" = []
        self.vertex_stack: "list[list[float]]" = []
        self.index_stack: "list[int]" = []
        self._door_material_index: "int | None" = None
        pass
    
    def add_quad(self, a, b, c, d, invert_normals = False):
        i = len(self.vertex_stack)
        if invert_normals:
            self.index_stack.extend([i + 0, i + 3, i + 2, i + 2, i + 1, i + 0])
        else:
            self.index_stack.extend([i + 0, i + 1, i + 2, i + 2, i + 3, i + 0])
        self.vertex_stack.extend([a, b, c, d])

    def add_cube(self, x1: float, y1: float, x2: float, y2: float, z1: float, z2: float):
         # Z-
        self.add_quad(
            [x1, y1, z1],
            [x1, y2, z1],
            [x2, y2, z1],
            [x2, y1, z1],
            invert_normals=True,
        )

        # Z+
        self.add_quad(
            [x1, y1, z2],
            [x1, y2, z2],
            [x2, y2, z2],
            [x2, y1, z2],
        )

        # Y+
        self.add_quad(
            [x1, y1, z1],
            [x2, y1, z1],
            [x2, y1, z2],
            [x1, y1, z2],
            invert_normals=True,
        )

        # Y-
        self.add_quad(
            [x1, y2, z1],
            [x2, y2, z1],
            [x2, y2, z2],
            [x1, y2, z2],
        )

        # X+
        self.add_quad(
            [x1, y1, z1],
            [x1, y2, z1],
            [x1, y2, z2],
            [x1, y1, z2],
        )

        # X-
        self.add_quad(
            [x2, y1, z1],
            [x2, y2, z1],
            [x2, y2, z2],
            [x2, y1, z2],
            invert_normals=True,
        )

    def create_mesh(self, name: "str | list[str]", indices = None, vertices = None, invert_normals = False):
        if type(name) == str:
            names = [name]
        else:
            names = name
            name = names[0]

        if indices is None:
            indices = self.index_stack
            self.index_stack = []
            
        if vertices is None:
            vertices = self.vertex_stack
            self.vertex_stack = []

        # 1. Define the Mesh Data (Vertices and Indices)

        # Example: A simple square (two triangles)
        # Vertices (x, y, z) - float32 is common for positions
        # Note: glTF uses a right-handed coordinate system, typically Y-up.
        vertices = np.array(
            vertices,
            dtype=np.float32,
        )

        # Indices (to form two triangles: 0-1-2 and 1-3-2) - uint16 is common for indices

        indices = np.array(
            indices,
            dtype=np.uint16,
        )

        if invert_normals:
            indices = indices[::-1]

        # 2. Reorder the columns to (0, 2, 1) using fancy indexing
        # This selects column 0 (X), the modified column 2 (-Z), and column 1 (Y)
        vertices = vertices[:, [0, 2, 1]]

        # 2. Convert Data to Binary Buffers

        # Combine all data into a single binary buffer for efficiency
        # gltflib handles the packing into a single bytes object.
        binary_blob = vertices.tobytes() + indices.tobytes()

        resource_name = name + ".bin"
        resource = FileResource(resource_name, data=binary_blob)
        self.gltf_resources.append(resource)

        buffer = create(self.gltf_buffers, Buffer(byteLength=len(binary_blob), uri=resource_name))

        # 3. Define Buffer Views

        # A BufferView describes a segment of a Buffer.

        # For Vertices (Position data)
        vertex_byte_offset = 0
        vertex_byte_length = vertices.nbytes
        vertex_buffer_id = create(self.gltf_buffer_views,  BufferView(
            buffer=buffer,
            byteOffset=vertex_byte_offset,
            byteLength=vertex_byte_length,
            # target=34962,  # ARRAY_BUFFER (Optional, but good practice)
        ))

        # For Indices
        index_byte_offset = vertex_byte_length  # Starts after the vertices
        index_byte_length = indices.nbytes
        index_buffer_id = create(self.gltf_buffer_views, BufferView(
            buffer=buffer,
            byteOffset=index_byte_offset,
            byteLength=index_byte_length,
            # target=34963,  # ELEMENT_ARRAY_BUFFER (Optional, but good practice)
        ))

        # 4. Define Accessors

        # Accessors define how to interpret the data in a BufferView.

        # Accessor for Positions (Vertices)
        position_accessor = create(self.gltf_accessors, Accessor(
            bufferView=vertex_buffer_id,
            byteOffset=0,
            componentType=ComponentType.FLOAT,
            count=len(vertices),
            type=AccessorType.VEC3.value,
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist(),
        ))

        # Accessor for Indices
        index_accessor = create(self.gltf_accessors, Accessor(
            bufferView=index_buffer_id,
            byteOffset=0,
            componentType=ComponentType.UNSIGNED_SHORT,  # Corresponds to np.uint16
            count=len(indices),
            type=AccessorType.SCALAR.value,  # Single value per index
        ))

        # Determine material (brown for doors)
        material_index = None
        if isinstance(name, str) and name.startswith("Door_"):
            # Reuse a single brown material for all doors
            if self._door_material_index is None:
                pbr = PBRMetallicRoughness(
                    baseColorFactor=[0.55, 0.27, 0.07, 1.0],  # brown RGBA
                )
                material = Material(
                    name="DoorBrown",
                    pbrMetallicRoughness=pbr,
                )
                self._door_material_index = create(self.gltf_materials, material)
            material_index = self._door_material_index

        # 5. Build the Mesh Primitive

        # A Primitive is the actual drawing geometry (e.g., a set of triangles).
        primitive = Primitive(
            attributes=Attributes(POSITION=position_accessor),  # POSITION uses the first accessor (index 0)
            indices=index_accessor,  # Indices use the second accessor (index 1)
            mode=PrimitiveMode.TRIANGLES.value,
            material=material_index,
        )

        mesh = create(self.gltf_meshes, Mesh(primitives=[primitive], name=name))
        for node_name in names:
            create(self.gltf_nodes, Node(mesh=mesh, name=node_name))

    def build(self):
        model = GLTFModel(
            asset=Asset(version='2.0'),
            scenes=[Scene(nodes=list(range(len(self.gltf_nodes))))],
            nodes=self.gltf_nodes,
            meshes=self.gltf_meshes,
            buffers=self.gltf_buffers,
            bufferViews=self.gltf_buffer_views,
            accessors=self.gltf_accessors,
            materials=self.gltf_materials,
        )

        return GLTF(model=model, resources=self.gltf_resources)