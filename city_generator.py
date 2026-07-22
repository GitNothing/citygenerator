"""Procedural city generator for Blender 3.6+ / 4.x.

Run this file from Blender's Scripting workspace.  It clears the current scene,
then creates a road grid, sidewalks, a center-weighted skyline, medium apartment
buildings, low-rise perimeter buildings, windows, rooftop details, lighting, and
a presentation camera.

The most useful controls are collected in the CONFIG section immediately below.
"""

import math
import random

import bpy
from mathutils import Vector


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

SEED = 27
BLOCKS_X = 20
BLOCKS_Y = 20
BLOCK_SIZE = 32.0
ROAD_WIDTH = 10.0
LOTS_PER_BLOCK = 4
CITY_MARGIN = 38.0

SIDEWALK_HEIGHT = 0.32
FLOOR_HEIGHT = 3.25
WINDOW_HEIGHT = 1.75
WINDOW_SPACING = 2.45
WINDOW_EMISSION_STRENGTH = 1.7

OUTLYING_TOWER_CHANCE = 0.06
OUTLYING_TOWER_MIN_HEIGHT = 30.0
OUTLYING_TOWER_MAX_HEIGHT = 48.0

# Radial zones are measured from the city center.  The outer corners can have a
# value above 1.0, which intentionally keeps the perimeter low.
TOWER_ZONE_RADIUS = 0.42
APARTMENT_ZONE_RADIUS = 0.88

ADD_LANE_MARKINGS = True
ADD_CAMERA_AND_LIGHTS = True


# -----------------------------------------------------------------------------
# LOW-LEVEL HELPERS
# -----------------------------------------------------------------------------


def clear_scene():
    """Remove the current scene contents so rerunning the script stays clean."""
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.materials,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def make_collection(name):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def make_material(name, color, roughness=0.55, metallic=0.0, emission=None):
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True

    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader:
        base = shader.inputs.get("Base Color")
        if base:
            base.default_value = color
        rough = shader.inputs.get("Roughness")
        if rough:
            rough.default_value = roughness
        metal = shader.inputs.get("Metallic")
        if metal:
            metal.default_value = metallic

        if emission is not None:
            emission_color, emission_strength = emission
            emission_input = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
            strength_input = shader.inputs.get("Emission Strength")
            if emission_input:
                emission_input.default_value = emission_color
            if strength_input:
                strength_input.default_value = emission_strength

    return material


def create_box(name, dimensions, location, material, collection, bevel=0.0):
    """Create a box with dimensions baked into its mesh (good bevel behavior)."""
    dx, dy, dz = (value * 0.5 for value in dimensions)
    vertices = [
        (-dx, -dy, -dz),
        (dx, -dy, -dz),
        (dx, dy, -dz),
        (-dx, dy, -dz),
        (-dx, -dy, dz),
        (dx, -dy, dz),
        (dx, dy, dz),
        (-dx, dy, dz),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    collection.objects.link(obj)

    if bevel > 0.0:
        modifier = obj.modifiers.new(name="Soft_Edges", type="BEVEL")
        modifier.width = bevel
        modifier.segments = 2

    return obj


def add_quad(vertices, faces, material_indices, corners, material_index):
    start = len(vertices)
    vertices.extend(corners)
    faces.append((start, start + 1, start + 2, start + 3))
    material_indices.append(material_index)


def create_window_panels(
    name,
    center_x,
    center_y,
    width,
    depth,
    bottom_z,
    height,
    window_dark,
    window_lit,
    collection,
    rng,
    residential=False,
):
    """Build all four window façades as one lightweight mesh object."""
    vertices = []
    faces = []
    material_indices = []

    floors = max(2, int(height / FLOOR_HEIGHT))
    edge_margin = 1.15
    panel_offset = 0.025
    lit_probability = 0.28 if residential else 0.20

    def centers_along(length):
        usable = max(0.1, length - edge_margin * 2.0)
        count = max(1, int(usable / WINDOW_SPACING))
        spacing = usable / count
        return [(-usable * 0.5) + spacing * (i + 0.5) for i in range(count)], spacing

    x_centers, x_spacing = centers_along(width)
    y_centers, y_spacing = centers_along(depth)
    win_width_x = min(1.55, x_spacing * 0.66)
    win_width_y = min(1.55, y_spacing * 0.66)

    for floor in range(floors):
        z_center = bottom_z + (floor + 0.56) * (height / floors)
        half_h = min(WINDOW_HEIGHT, (height / floors) * 0.58) * 0.5

        # Front and back: quads lie in X/Z.
        for local_x in x_centers:
            x = center_x + local_x
            half_w = win_width_x * 0.5
            for side in (-1.0, 1.0):
                y = center_y + side * (depth * 0.5 + panel_offset)
                corners = [
                    (x - half_w, y, z_center - half_h),
                    (x + half_w, y, z_center - half_h),
                    (x + half_w, y, z_center + half_h),
                    (x - half_w, y, z_center + half_h),
                ]
                add_quad(vertices, faces, material_indices, corners, int(rng.random() < lit_probability))

        # Left and right: quads lie in Y/Z.
        for local_y in y_centers:
            y = center_y + local_y
            half_w = win_width_y * 0.5
            for side in (-1.0, 1.0):
                x = center_x + side * (width * 0.5 + panel_offset)
                corners = [
                    (x, y - half_w, z_center - half_h),
                    (x, y + half_w, z_center - half_h),
                    (x, y + half_w, z_center + half_h),
                    (x, y - half_w, z_center + half_h),
                ]
                add_quad(vertices, faces, material_indices, corners, int(rng.random() < lit_probability))

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(window_dark)
    mesh.materials.append(window_lit)
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.material_index = material_index
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj, floors


def point_camera_at(camera, target):
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


# -----------------------------------------------------------------------------
# CITY COMPONENTS
# -----------------------------------------------------------------------------


def build_materials():
    materials = {
        "grass": make_material("Ground_Grass", (0.055, 0.115, 0.055, 1.0), roughness=0.92),
        "road": make_material("Road_Asphalt", (0.026, 0.030, 0.036, 1.0), roughness=0.88),
        "sidewalk": make_material("Sidewalk", (0.32, 0.34, 0.35, 1.0), roughness=0.82),
        "lane": make_material("Lane_Marking", (0.95, 0.67, 0.08, 1.0), roughness=0.58),
        "roof": make_material("Roof_Details", (0.12, 0.13, 0.14, 1.0), roughness=0.75),
        "window_dark": make_material(
            "Windows_Dark", (0.015, 0.045, 0.072, 1.0), roughness=0.17, metallic=0.18
        ),
        "window_lit": make_material(
            "Windows_Lit",
            (0.54, 0.30, 0.08, 1.0),
            roughness=0.2,
            emission=((1.0, 0.45, 0.10, 1.0), WINDOW_EMISSION_STRENGTH),
        ),
    }

    materials["low"] = [
        make_material("Low_Brick", (0.31, 0.12, 0.075, 1.0), roughness=0.78),
        make_material("Low_Sandstone", (0.46, 0.34, 0.22, 1.0), roughness=0.80),
        make_material("Low_Concrete", (0.34, 0.36, 0.37, 1.0), roughness=0.86),
    ]
    materials["apartment"] = [
        make_material("Apartment_Red", (0.36, 0.105, 0.065, 1.0), roughness=0.72),
        make_material("Apartment_Tan", (0.53, 0.39, 0.27, 1.0), roughness=0.76),
        make_material("Apartment_White", (0.57, 0.59, 0.58, 1.0), roughness=0.70),
    ]
    materials["tower"] = [
        make_material("Tower_Blue", (0.075, 0.16, 0.22, 1.0), roughness=0.28, metallic=0.12),
        make_material("Tower_Slate", (0.12, 0.15, 0.18, 1.0), roughness=0.34, metallic=0.16),
        make_material("Tower_Steel", (0.20, 0.23, 0.25, 1.0), roughness=0.31, metallic=0.26),
    ]
    return materials


def create_roads_and_blocks(materials, collections):
    pitch = BLOCK_SIZE + ROAD_WIDTH
    city_width = BLOCKS_X * pitch + ROAD_WIDTH
    city_depth = BLOCKS_Y * pitch + ROAD_WIDTH

    create_box(
        "City_Ground",
        (city_width + CITY_MARGIN * 2.0, city_depth + CITY_MARGIN * 2.0, 0.3),
        (0.0, 0.0, -0.18),
        materials["grass"],
        collections["ground"],
    )

    # Road centers sit on every block boundary, including the outside boundary.
    for road_x in range(BLOCKS_X + 1):
        x = (road_x - BLOCKS_X * 0.5) * pitch
        create_box(
            f"Road_V_{road_x:02d}",
            (ROAD_WIDTH, city_depth, 0.12),
            (x, 0.0, 0.0),
            materials["road"],
            collections["roads"],
        )

    for road_y in range(BLOCKS_Y + 1):
        y = (road_y - BLOCKS_Y * 0.5) * pitch
        create_box(
            f"Road_H_{road_y:02d}",
            (city_width, ROAD_WIDTH, 0.12),
            (0.0, y, 0.0),
            materials["road"],
            collections["roads"],
        )

    for block_x in range(BLOCKS_X):
        x = (block_x - (BLOCKS_X - 1) * 0.5) * pitch
        for block_y in range(BLOCKS_Y):
            y = (block_y - (BLOCKS_Y - 1) * 0.5) * pitch
            create_box(
                f"Sidewalk_{block_x:02d}_{block_y:02d}",
                (BLOCK_SIZE, BLOCK_SIZE, SIDEWALK_HEIGHT),
                (x, y, SIDEWALK_HEIGHT * 0.5),
                materials["sidewalk"],
                collections["sidewalks"],
                bevel=0.12,
            )

    if ADD_LANE_MARKINGS:
        dash_length = 5.0
        dash_gap = 5.0
        step = dash_length + dash_gap
        dash_count_y = int(city_depth / step)
        dash_count_x = int(city_width / step)

        for road_x in range(BLOCKS_X + 1):
            x = (road_x - BLOCKS_X * 0.5) * pitch
            for dash in range(dash_count_y):
                y = -city_depth * 0.5 + step * (dash + 0.5)
                create_box(
                    f"Lane_V_{road_x:02d}_{dash:03d}",
                    (0.22, dash_length, 0.025),
                    (x, y, 0.073),
                    materials["lane"],
                    collections["markings"],
                )

        for road_y in range(BLOCKS_Y + 1):
            y = (road_y - BLOCKS_Y * 0.5) * pitch
            for dash in range(dash_count_x):
                x = -city_width * 0.5 + step * (dash + 0.5)
                create_box(
                    f"Lane_H_{road_y:02d}_{dash:03d}",
                    (dash_length, 0.22, 0.025),
                    (x, y, 0.074),
                    materials["lane"],
                    collections["markings"],
                )

    return pitch, city_width, city_depth


def add_rooftop_details(name, building_type, x, y, width, depth, top_z, materials, collection, rng):
    if building_type == "TOWER":
        cap_width = width * rng.uniform(0.38, 0.60)
        cap_depth = depth * rng.uniform(0.38, 0.60)
        cap_height = rng.uniform(1.1, 2.2)
        create_box(
            name + "_Mechanical_Penthouse",
            (cap_width, cap_depth, cap_height),
            (x, y, top_z + cap_height * 0.5),
            materials["roof"],
            collection,
            bevel=0.12,
        )

    elif building_type == "APARTMENT":
        cap_height = 0.45
        create_box(
            name + "_Roof_Parapet",
            (width * 0.94, depth * 0.94, cap_height),
            (x, y, top_z + cap_height * 0.5),
            materials["roof"],
            collection,
            bevel=0.08,
        )
        if rng.random() < 0.78:
            unit_width = min(3.2, width * 0.28)
            unit_depth = min(2.7, depth * 0.25)
            unit_height = rng.uniform(0.65, 1.0)
            create_box(
                name + "_HVAC",
                (unit_width, unit_depth, unit_height),
                (x + width * 0.20, y - depth * 0.16, top_z + cap_height + unit_height * 0.5),
                materials["roof"],
                collection,
                bevel=0.06,
            )

    else:
        cap_height = 0.28
        create_box(
            name + "_Roof_Edge",
            (width * 0.96, depth * 0.96, cap_height),
            (x, y, top_z + cap_height * 0.5),
            materials["roof"],
            collection,
            bevel=0.05,
        )


def create_building(
    name,
    building_type,
    x,
    y,
    width,
    depth,
    height,
    radial_distance,
    materials,
    collections,
    rng,
):
    bottom_z = SIDEWALK_HEIGHT
    palette_key = building_type.lower()
    body_material = rng.choice(materials[palette_key])
    bevel = 0.24 if building_type == "TOWER" else 0.16

    body = create_box(
        name + "_Body",
        (width, depth, height),
        (x, y, bottom_z + height * 0.5),
        body_material,
        collections["buildings"],
        bevel=bevel,
    )

    _, floors = create_window_panels(
        name + "_Windows",
        x,
        y,
        width,
        depth,
        bottom_z,
        height,
        materials["window_dark"],
        materials["window_lit"],
        collections["windows"],
        rng,
        residential=(building_type == "APARTMENT"),
    )

    body["building_type"] = building_type
    body["floors"] = floors
    body["height_m"] = round(height, 2)
    body["distance_from_center"] = round(radial_distance, 3)

    add_rooftop_details(
        name,
        building_type,
        x,
        y,
        width,
        depth,
        bottom_z + height,
        materials,
        collections["details"],
        rng,
    )
    return floors


def choose_building_type_and_height(radial_distance, rng):
    # Occasionally place a tall building outside the central tower zone.
    if (
        radial_distance >= TOWER_ZONE_RADIUS
        and rng.random() < OUTLYING_TOWER_CHANCE
    ):
        height = rng.uniform(
            OUTLYING_TOWER_MIN_HEIGHT,
            OUTLYING_TOWER_MAX_HEIGHT,
        )
        return "TOWER", height
    """Use radial zoning so the skyline falls toward the perimeter."""
    if radial_distance < TOWER_ZONE_RADIUS:
        building_type = "TOWER"
        center_strength = 1.0 - radial_distance / TOWER_ZONE_RADIUS
        height = 43.0 + 40.0 * center_strength + rng.uniform(-3.0, 9.0)
    elif radial_distance < APARTMENT_ZONE_RADIUS:
        # Most of the middle ring is residential-scale apartment stock.
        building_type = "APARTMENT" if rng.random() < 0.78 else "LOW"
        if building_type == "APARTMENT":
            ring_strength = 1.0 - (
                (radial_distance - TOWER_ZONE_RADIUS)
                / (APARTMENT_ZONE_RADIUS - TOWER_ZONE_RADIUS)
            )
            height = 19.0 + 14.0 * ring_strength + rng.uniform(-1.5, 4.0)
        else:
            height = rng.uniform(10.0, 17.0)
    else:
        building_type = "LOW"
        height = rng.uniform(6.5, 13.0)

    return building_type, max(6.0, height)


def create_city_buildings(pitch, materials, collections, rng):
    max_center_x = max(pitch, (BLOCKS_X - 1) * pitch * 0.5)
    max_center_y = max(pitch, (BLOCKS_Y - 1) * pitch * 0.5)
    half_lot = BLOCK_SIZE * 0.25
    lot_centers = [
        (-half_lot, -half_lot),
        (half_lot, -half_lot),
        (-half_lot, half_lot),
        (half_lot, half_lot),
    ]

    stats = {"TOWER": 0, "APARTMENT": 0, "LOW": 0, "floors": 0}

    for block_x in range(BLOCKS_X):
        block_center_x = (block_x - (BLOCKS_X - 1) * 0.5) * pitch
        for block_y in range(BLOCKS_Y):
            block_center_y = (block_y - (BLOCKS_Y - 1) * 0.5) * pitch

            # Reserve the exact middle block for a single landmark skyscraper.
            is_center_block = (
                BLOCKS_X % 2 == 1
                and BLOCKS_Y % 2 == 1
                and block_x == BLOCKS_X // 2
                and block_y == BLOCKS_Y // 2
            )
            if is_center_block:
                floors = create_building(
                    "Central_Landmark",
                    "TOWER",
                    block_center_x,
                    block_center_y,
                    BLOCK_SIZE * 0.58,
                    BLOCK_SIZE * 0.58,
                    94.0,
                    0.0,
                    materials,
                    collections,
                    rng,
                )
                stats["TOWER"] += 1
                stats["floors"] += floors

                antenna_height = 13.0
                create_box(
                    "Central_Landmark_Antenna",
                    (0.42, 0.42, antenna_height),
                    (block_center_x, block_center_y, SIDEWALK_HEIGHT + 94.0 + 2.0 + antenna_height * 0.5),
                    materials["roof"],
                    collections["details"],
                )
                continue

            shuffled_lots = lot_centers[:]
            rng.shuffle(shuffled_lots)

            block_radius = math.sqrt(
                (block_center_x / max_center_x) ** 2
                + (block_center_y / max_center_y) ** 2
            )
            if block_radius < TOWER_ZONE_RADIUS:
                building_count = 2
            elif block_radius < APARTMENT_ZONE_RADIUS:
                building_count = 3
            else:
                building_count = 3 if rng.random() < 0.78 else 2

            for lot_index, (offset_x, offset_y) in enumerate(shuffled_lots[:building_count]):
                jitter = 0.75
                x = block_center_x + offset_x + rng.uniform(-jitter, jitter)
                y = block_center_y + offset_y + rng.uniform(-jitter, jitter)
                radial_distance = math.sqrt(
                    (x / max_center_x) ** 2 + (y / max_center_y) ** 2
                )

                building_type, height = choose_building_type_and_height(radial_distance, rng)

                # A 2x2 lot pattern leaves clear gaps between neighboring buildings.
                max_footprint = BLOCK_SIZE * 0.42
                if building_type == "APARTMENT":
                    width = rng.uniform(max_footprint * 0.78, max_footprint)
                    depth = rng.uniform(max_footprint * 0.78, max_footprint)
                elif building_type == "TOWER":
                    width = rng.uniform(max_footprint * 0.72, max_footprint * 0.94)
                    depth = rng.uniform(max_footprint * 0.72, max_footprint * 0.94)
                else:
                    width = rng.uniform(max_footprint * 0.58, max_footprint * 0.92)
                    depth = rng.uniform(max_footprint * 0.58, max_footprint * 0.92)

                name = f"{building_type.title()}_{block_x:02d}_{block_y:02d}_{lot_index:02d}"
                floors = create_building(
                    name,
                    building_type,
                    x,
                    y,
                    width,
                    depth,
                    height,
                    radial_distance,
                    materials,
                    collections,
                    rng,
                )
                stats[building_type] += 1
                stats["floors"] += floors

    return stats


def add_camera_and_lighting(city_width, city_depth, collections):
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("City_World")
        bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.075, 0.11, 0.17, 1.0)
        background.inputs["Strength"].default_value = 0.48

    sun_data = bpy.data.lights.new(name="City_Sun", type="SUN")
    sun_data.energy = 3.2
    sun_data.angle = math.radians(18.0)
    sun = bpy.data.objects.new("City_Sun", sun_data)
    sun.rotation_euler = (
        math.radians(32.0),
        math.radians(-18.0),
        math.radians(-38.0),
    )
    collections["lighting"].objects.link(sun)

    area_data = bpy.data.lights.new(name="City_Fill", type="AREA")
    area_data.energy = 1800.0
    area_data.shape = "DISK"
    area_data.size = 110.0
    area = bpy.data.objects.new("City_Fill", area_data)
    area.location = (0.0, 0.0, 145.0)
    collections["lighting"].objects.link(area)
    area.rotation_euler = (0.0, 0.0, 0.0)

    camera_data = bpy.data.cameras.new("City_Camera")
    camera = bpy.data.objects.new("City_Camera", camera_data)
    camera.location = (
        city_width * 0.67,
        -city_depth * 0.78,
        max(city_width, city_depth) * 0.54,
    )
    camera_data.lens = 52.0
    camera_data.sensor_width = 36.0
    point_camera_at(camera, (0.0, 0.0, 25.0))
    collections["camera"].objects.link(camera)
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100

    # These settings work in both Blender 3.6 and 4.x.
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = "//procedural_city.png"
    # Color-management look names changed between Blender 3.6 and 4.x.
    try:
        if "AgX" in scene.view_settings.view_transform:
            scene.view_settings.look = "AgX - Medium High Contrast"
        else:
            scene.view_settings.look = "Medium High Contrast"
    except TypeError:
        # A custom OCIO configuration may expose different look names.
        pass


def generate_city():
    clear_scene()
    rng = random.Random(SEED)

    collections = {
        "ground": make_collection("00_Ground"),
        "roads": make_collection("01_Roads"),
        "markings": make_collection("02_Lane_Markings"),
        "sidewalks": make_collection("03_Sidewalks"),
        "buildings": make_collection("04_Buildings"),
        "windows": make_collection("05_Windows"),
        "details": make_collection("06_Rooftop_Details"),
        "lighting": make_collection("07_Lighting"),
        "camera": make_collection("08_Camera"),
    }
    materials = build_materials()
    pitch, city_width, city_depth = create_roads_and_blocks(materials, collections)
    stats = create_city_buildings(pitch, materials, collections, rng)

    if ADD_CAMERA_AND_LIGHTS:
        add_camera_and_lighting(city_width, city_depth, collections)

    scene = bpy.context.scene
    scene["city_seed"] = SEED
    scene["tower_count"] = stats["TOWER"]
    scene["apartment_count"] = stats["APARTMENT"]
    scene["low_rise_count"] = stats["LOW"]

    total = stats["TOWER"] + stats["APARTMENT"] + stats["LOW"]
    print("\nProcedural city complete")
    print(f"  Buildings: {total}")
    print(f"  Towers: {stats['TOWER']}")
    print(f"  Apartments: {stats['APARTMENT']}")
    print(f"  Low-rise: {stats['LOW']}")
    print(f"  Total modeled floors: {stats['floors']}")
    print("  Press Numpad 0 for the generated camera view.")


if __name__ == "__main__":
    generate_city()
