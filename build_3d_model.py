from bisect import bisect_left
from json import loads
from math import inf, cos, sin, radians
from sys import argv
from typing import Any, Callable
from dataclasses import dataclass, field
import numpy as np
from MeshBuilder import MeshBuilder

"""
Directions:
  0 -> UP
  1 -> RIGHT
  2 -> DOWN
  3 -> LEFT
"""

@dataclass(eq=False)
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float

    type: str

    group: "set[Wall] | None" = None

    def get_width(self):
        return self.x2 - self.x1

    def set_width(self, value: float):
        if self.group is not None:
            for wall in self.group:
                wall.x2 = wall.x1 + value
        else:
            self.x2 = self.x1 + value

    def get_height(self):
        return self.y2 - self.y1

    def set_height(self, value: float):
        if self.group is not None:
            for wall in self.group:
                wall.y2 = wall.y1 + value
        else:
            self.y2 = self.y1 + value

    def is_horizontal(self):
        return self.get_width() > self.get_height()

    def translate(self, x: float, y: float):
        if self.group is not None:
            for wall in self.group:
                wall._translateSelf(x, y)
        else:
            self._translateSelf(x, y)
    
    def _translateSelf(self, x: float, y: float):
        self.x1 += x
        self.x2 += x
        self.y1 += y
        self.y2 += y

    def normalize(self, normalizer: float):
        self.x1 *= normalizer
        self.x2 *= normalizer
        self.y1 *= normalizer
        self.y2 *= normalizer

    def link(self, other: "Wall"):
        # If both walls are not in a group, create a new group with both of them
        if self.group is None and other.group is None:
            self.group = other.group = {self, other}
            return

        # If both walls are grouped, merge the group
        if self.group is not None and other.group is not None:

            new_list = {*self.group, *other.group}
            for wall in self.group:
                wall.group = new_list

            for wall in other.group:
                wall.group = new_list
        
        # If other is in a group add self to it
        if other.group is not None:
            self.group = other.group
            other.group.add(self)
            return

        # If self is in a group add other to it
        if self.group is not None:
            other.group = self.group
            self.group.add(other)
            return
        
        # Unreachable
        assert False
    
    def get_point(self, direction: int):
        if direction == 0:
            return np.array(((self.x1 + self.x2) / 2, self.y1), dtype=np.float)

        if direction == 1:
            return np.array((self.x2, (self.y1 + self.y2) / 2), dtype=np.float)

        if direction == 2:
            return np.array(((self.x1 + self.x2) / 2, self.y2), dtype=np.float)

        if direction == 3:
            return np.array((self.x1, (self.y1 + self.y2) / 2), dtype=np.float)
        
        # Illegal direction
        assert False
        

@dataclass
class Socket:
    wall: Wall
    direction: int
    original_position: Any = field(init=False)
    tolerance: float

    def __post_init__(self):
        self.original_position = self.wall.get_point(self.direction)

    @property
    def position(self):
        return self.wall.get_point(self.direction)

    def get_opposite(self):
        return (self.direction + 2) % 4
    
    def is_horizontal(self):
        return self.direction == 1 or self.direction == 3

def align_walls(walls: "list[Wall]"):
    sockets: "dict[int, list[Socket]]" = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    for wall in walls:
        horizontal = wall.is_horizontal()
        vertical = not horizontal

        # If the wall is close enough to a square, consider is horizontal and vertical at once
        if abs((wall.get_height() - wall.get_width()) / (wall.get_height() + wall.get_width())) < 0.15:
            horizontal = True
            vertical = True

        if horizontal:
            tolerance = wall.get_height()
            sockets[1].append(Socket(wall, 1, tolerance))
            sockets[3].append(Socket(wall, 3, tolerance))

        if vertical:
            tolerance = wall.get_width()
            sockets[0].append(Socket(wall, 0, tolerance))
            sockets[2].append(Socket(wall, 2, tolerance))
    
    for direction in [0, 1]:
        direction_sockets = sockets[direction]
        j = 0
        matches = 0

        while j < len(direction_sockets):
            socket = direction_sockets[j]
            j += 1
            
            opposite_sockets = sockets[socket.get_opposite()]

            # Find socket with opposite direction that is close enough
            for opposite_socket in opposite_sockets:
                distance = np.linalg.norm(socket.original_position - opposite_socket.original_position)

                if socket.is_horizontal():
                    distance += abs(socket.wall.get_height() - opposite_socket.wall.get_height())
                else:
                    distance += abs(socket.wall.get_width() - opposite_socket.wall.get_width())

                tolerance = min(socket.tolerance, opposite_socket.tolerance)
                if distance <= tolerance:
                    break
            else:
                opposite_socket = None
            
            if opposite_socket is None:
                continue
            
            matches += 1

            # Remove the matched sockets
            direction_sockets.remove(socket)
            opposite_sockets.remove(opposite_socket)

            # Rollback iteration to account for removed element
            j -= 1

            # Unify the thickness
            if socket.is_horizontal():
                value = (socket.wall.get_height() + opposite_socket.wall.get_height()) / 2
                socket.wall.set_height(value)
                opposite_socket.wall.set_height(value)
            else:
                value = (socket.wall.get_width() + opposite_socket.wall.get_width()) / 2
                socket.wall.set_width(value)
                opposite_socket.wall.set_width(value)

            # Offset for the opposite socket to be aligned with this
            center = (opposite_socket.position + socket.position) / 2

            correction = (socket.position - center) * -1
            socket.wall.translate(*correction)

            correction = (opposite_socket.position - center) * -1
            opposite_socket.wall.translate(*correction)

            # Join the two walls together
            opposite_socket.wall.link(socket.wall)

        print(f"For direction {direction} aligned {matches} pairs")

def walls_from_json(data: dict):
    walls: "list[Wall]" = []

    for i, point in enumerate(data["points"]):
        walls.append(Wall(
            point["x1"], # type: ignore
            point["y1"],
            point["x2"],
            point["y2"],
            data["classes"][i]["name"]
        ))
    
    return walls

def walls_to_json(walls: "list[Wall]"):
    points = []

    for wall in walls:
        points.append({
            "x1": wall.x1,
            "y1": wall.y1,
            "x2": wall.x2,
            "y2": wall.y2,
        })

    return points

def build_geometry(walls: "list[Wall]"):
    pass

def find_rooms(walls: "list[Wall]", tolerance: float, sample_image: "Callable[[float, float], bool] | None" = None):
    # Discretize the floorplan into a grid based on wall coordinates
    x_grid: "list[float]" = []
    y_grid: "list[float]" = []

    grid_tolerance = inf
    for wall in walls:
        grid_tolerance = min(grid_tolerance, wall.get_width(), wall.get_height())

    grid_tolerance *= 0.75

    def push_grid_line(grid: "list[float]", position: float):
        index = bisect_left(grid, position)
        neighbour_min = grid[index - 1] if 0 <= index - 1 < len(grid) else None
        if neighbour_min is not None and abs(position - neighbour_min) < grid_tolerance:
            return neighbour_min

        neighbour_max = grid[index] if 0 <= index < len(grid) else None
        if neighbour_max is not None and abs(position - neighbour_max) < grid_tolerance:
            return neighbour_max
        
        grid.insert(index, position)
        return position

    average_wall_thickness_sum = 0

    # Create the grid based on the walls while also snapping their boundaries to the grid
    for wall in walls:
        # If some walls are too thin, they will be flattened, increase their thickness so that does not happen
        if wall.x2 - wall.x1 < grid_tolerance * 2:
            center = (wall.x1 + wall.x2) * 0.5
            wall.x1 = center - grid_tolerance * 1.2
            wall.x2 = center + grid_tolerance * 1.2

        if wall.y2 - wall.y1 < grid_tolerance * 2:
            center = (wall.y1 + wall.y2) * 0.5
            wall.y1 = center - grid_tolerance * 1.2
            wall.y2 = center + grid_tolerance * 1.2

        wall.x1 = push_grid_line(x_grid, wall.x1)
        wall.x2 = push_grid_line(x_grid, wall.x2)

        wall.y1 = push_grid_line(y_grid, wall.y1)
        wall.y2 = push_grid_line(y_grid, wall.y2)

        average_wall_thickness_sum += wall.get_height() if wall.is_horizontal() else wall.get_width()

    # Create virtual walls on the edges of the grid for handling the model sometimes not detecting boundary walls 
    average_wall_thickness = average_wall_thickness_sum / len(walls)

    x_start = x_grid[0]
    x_end = x_grid[-1]
    push_grid_line(x_grid, x_start + average_wall_thickness)
    push_grid_line(x_grid, x_start - average_wall_thickness)
    push_grid_line(x_grid, x_end + average_wall_thickness)
    push_grid_line(x_grid, x_end - average_wall_thickness)

    y_start = y_grid[0]
    y_end = y_grid[-1]
    push_grid_line(y_grid, y_start + average_wall_thickness)
    push_grid_line(y_grid, y_start - average_wall_thickness)
    push_grid_line(y_grid, y_end + average_wall_thickness)
    push_grid_line(y_grid, y_end - average_wall_thickness)
    
    # Create the grid. The values are so:
    # -1  => Unassigned empty cell
    #  0  => Wall
    # ≥ 1 => Assigned to a specific room
    width = len(x_grid) - 1
    height = len(y_grid) - 1
    tiles: "list[int]" = [-1] * (width * height)

    for y, (y1, y2) in enumerate(zip(y_grid, y_grid[1:])):
        for x, (x1, x2) in enumerate(zip(x_grid, x_grid[1:])):
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            for wall in walls:
                if wall.x1 < center_x < wall.x2 \
                    and wall.y1 < center_y < wall.y2:
                    tiles[x + y * width] = 0
                    break

    # Find missing walls by checking the image for every empty cell. By sampling pixels, if the cell is 75% black pixels, it's probably a wall.
    if sample_image is not None:
        for y, (y1, y2) in enumerate(zip(y_grid, y_grid[1:])):
            for x, (x1, x2) in enumerate(zip(x_grid, x_grid[1:])):
                if tiles[x + y * width] == 0:
                    continue

                black = 0
                count = 0

                for sx in range(1, 5):
                    for sy in range(1, 5):
                        xi = x1 + (x2 - x1) * (sx / 5)
                        yi = y1 + (y2 - y1) * (sy / 5)
                        is_white = sample_image(xi, yi)

                        count += 1
                        if not is_white:
                            black += 1

                if black / count < 0.75:
                    continue

                print(f"Fixing missing wall {(x1, y1, x2, y2)}")

                tiles[x + y * width] = 0
                walls.append(Wall(x1, y1, x2, y2, "wall")) # type: ignore
    
    # Fill in gaps between walls
    for y, (y1, y2) in enumerate(zip(y_grid, y_grid[1:])):
        for x, (x1, x2) in enumerate(zip(x_grid, x_grid[1:])):
            if x == 0 or x == width - 1 or y == 0 or y == height - 1 or tiles[x + y * width] == 0:
                continue 

            cell_width = x2 - x1
            cell_height = y2 - y1

            # Detect the presence of neighbouring walls
            is_gap_vertical = cell_height < tolerance * 3 and tiles[x + (y - 1) * width] == 0 and tiles[x + (y + 1) * width] == 0
            is_gap_horizontal = cell_width < tolerance * 3 and tiles[(x - 1) + y * width] == 0 and tiles[(x + 1) + y * width] == 0

            if not is_gap_horizontal and not is_gap_vertical:
                continue

            print(f"Fixing gap {(x1, y1, x2, y2)}")

            tiles[x + y * width] = 0
            walls.append(Wall(x1, y1, x2, y2, "wall")) # type: ignore
    
    room_id = 1
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for y in range(height):
        for x in range(width):
            index = x + y * width

            # Start floodfill if an unvisited free cell is found
            if tiles[index] == -1:
                stack = [(x, y)]
                tiles[index] = room_id

                while stack:
                    cx, cy = stack.pop()

                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy

                        # Check bounds
                        if 0 <= nx < width and 0 <= ny < height:
                            n_index = nx + ny * width

                            # Check if the neighbor is a free cell (-1)
                            if tiles[n_index] == -1:
                                tiles[n_index] = room_id
                                stack.append((nx, ny))

                room_id += 1

    # Create polygons out of the created rooms via greedy meshing. A room is ideally one rectangle,
    # but if the shape is more complex, we create a list of rectangles associated with each room.
    # The reason we start at 1 and stop at the penultimate tile is to ignore the cells created for virtual walls.
    occupied = [0] * len(tiles)
    room_meshes: "dict[int, list[tuple[float, float, float, float]]]" = {}
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            room_id = tiles[x + y * width]
            if room_id == 0 or occupied[x + y * width] != 0:
                continue

            iy = y + 1
            for iy in range(iy, height - 1):
                if tiles[x + iy * width] != room_id or occupied[x + iy * width] != 0:
                    break

            if iy == height:
                iy -= 1

            if y == iy:
                continue
            
            ix = x + 1
            for ix in range(ix, width - 1):
                failed = False

                for jy in range(y, iy):
                    if tiles[ix + jy * width] != room_id or occupied[ix + jy * width] != 0:
                        failed = True

                if failed:
                    break
                
            if ix == width:
                ix -= 1
            
            if x == ix:
                continue
            
            for jy in range(y, iy):
                for jx in range(x, ix):
                    occupied[jx + jy * width] = room_id
            
            room_meshes.setdefault(room_id, []).append((
                x_grid[x],
                y_grid[y],
                x_grid[ix],
                y_grid[iy],
            ))
    
    # Delete the rooms that are outside, which means they touch the edge of the grid. It may be the
    # case that a wall was not detected, exposing the inside to the outside, in which case the room
    # would be erroneously deleted. To handle this, only delete a room if it's small enough
    total_area = (x_end - x_start) * (y_end - y_start)
    def delete_outside_room(room_id: int):
        room_mesh = room_meshes[room_id]
        room_area = 0

        for (x1, y1, x2, y2) in room_mesh:
            room_area += (x2 - x1) * (y2 - y1)

        if room_area > total_area * 0.5:
            return

        del room_meshes[room_id]

    for x in range(width):
        room_id = tiles[x]
        if room_id != 0 and room_id in room_meshes:
            delete_outside_room(room_id)

        room_id = tiles[x + (width * (height - 1))]
        if room_id != 0 and room_id in room_meshes:
            delete_outside_room(room_id)

    for y in range(height):
        room_id = tiles[y * width]
        if room_id != 0 and room_id in room_meshes:
            delete_outside_room(room_id)

        room_id = tiles[y * width + width - 1]
        if room_id != 0 and room_id in room_meshes:
            delete_outside_room(room_id)
            
    print(f"Room grid: {width} x {height}, Room Count: {room_id - 1}")
    return room_meshes

def get_normalizer(data: dict):
    return 1 / (data["averageDoor"] / 0.8)

def build_3d_model(data: dict, sample_image: "Callable[[float, float], bool] | None" = None):
    walls = walls_from_json(data)
    align_walls(walls)

    normalizer = get_normalizer(data)
    for wall in walls:
        wall.normalize(normalizer)

    builder = MeshBuilder()
    rooms = find_rooms(walls, tolerance=0.1, sample_image=sample_image)
    data["points"] = walls_to_json(walls)

    # Create a single floor covering all ground from min to max x,y axes
    if walls:
        # Find the bounding box of all walls
        min_x = min(wall.x1 for wall in walls)
        max_x = max(wall.x2 for wall in walls)
        min_y = min(wall.y1 for wall in walls)
        max_y = max(wall.y2 for wall in walls)
        
        # Create a single floor quad covering all ground
        builder.add_quad(
            [min_x, min_y, 0],
            [min_x, max_y, 0],
            [max_x, max_y, 0],
            [max_x, min_y, 0],
        )
        
        builder.create_mesh(["Floor", "Ground"])

    height = 2.6

    for index, wall in enumerate(walls):
        x1 = wall.x1
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        def add_hinged_door(builder: MeshBuilder, x1: float, y1: float, x2: float, y2: float, z1: float, z2: float):
            """
            Create a door as an oriented box, rotated 45° around the hinge at (x1, y1),
            without scaling its size.
            """
            angle = radians(75.0)
            c = cos(angle)
            s = sin(angle)

            # Original rectangle corners in floor plane
            p0 = (x1, y1)          # hinge corner
            p1 = (x2, y1)
            p2 = (x2, y2)
            p3 = (x1, y2)

            def rotate_point(px: float, py: float):
                dx = px - x1
                dy = py - y1
                rx = x1 + dx * c - dy * s
                ry = y1 + dx * s + dy * c
                return rx, ry

            r0 = p0  # hinge stays fixed
            r1 = rotate_point(*p1)
            r2 = rotate_point(*p2)
            r3 = rotate_point(*p3)

            # Build 3D vertices (bottom z1, top z2)
            v0 = [r0[0], r0[1], z1]
            v1 = [r1[0], r1[1], z1]
            v2 = [r2[0], r2[1], z1]
            v3 = [r3[0], r3[1], z1]

            v4 = [r0[0], r0[1], z2]
            v5 = [r1[0], r1[1], z2]
            v6 = [r2[0], r2[1], z2]
            v7 = [r3[0], r3[1], z2]

            # Bottom face
            builder.add_quad(v0, v1, v2, v3, invert_normals=True)
            # Top face
            builder.add_quad(v4, v7, v6, v5)
            # Sides
            builder.add_quad(v0, v4, v5, v1)   # along hinge edge
            builder.add_quad(v1, v5, v6, v2)
            builder.add_quad(v2, v6, v7, v3)
            builder.add_quad(v3, v7, v4, v0)

        if wall.type == "window":
            builder.add_cube(x1, y1, x2, y2, 0, height / 3)
            builder.add_cube(x1, y1, x2, y2, height * 2/3, height)
        elif wall.type == "door":
            # Door opening: rotated 45° around the hinge at (x1, y1) without scaling
            # Door normally do not reach the ceiling, to achieve this, create we shorter doors and
            # then put wall above. Average height of doors is 200cm relative to ceiling height 260cm,
            # to keep this for any height use a ratio. The wall object is also create separately for
            # applying textures.
            tx1, ty1, tx2, ty2 = x1, y1, x2, y2
            dx = tx2 - tx1
            dy = ty2 - ty1

            # Make the door leaf thinner (1/3 of current thickness).
            # We keep the hinge edge fixed (at x1/y1 side) and shrink the smaller dimension.
            if dx >= dy:
                # Horizontal-ish door: thickness is along Y
                if dy > 0:
                    ty2 = ty1 + (dy / 3.0)
            else:
                # Vertical-ish door: thickness is along X
                if dx > 0:
                    tx2 = tx1 + (dx / 3.0)

            add_hinged_door(builder, tx1, ty1, tx2, ty2, 0, height * (10/13))
        else:
            builder.add_cube(x1, y1, x2, y2, 0, height)
        builder.create_mesh(f"{wall.type.capitalize()}_{index}")

        if wall.type == "door":
            # Wall above the door: keep original (unrotated) bounds
            builder.add_cube(x1, y1, x2, y2, height * (10/13), height)
            builder.create_mesh(f"Wall_{index}")
    
    return builder.build()

if __name__ == "__main__":
    with open(argv[1], 'rt') as file:
        content = file.read()

    data: dict = loads(content)
    gltf = build_3d_model(data)
    gltf.export(argv[1] + ".new.glb")

