"""
Utils for rendering games
"""
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


img_paths = glob.glob('/home/gbarbadillo/Desktop/luxai_icons/128px/*.png')
icons = {os.path.splitext(os.path.basename(img_path))[0]: plt.imread(img_path) for img_path in img_paths}

unit_number_to_name = {0: 'worker', 1: 'cart'}


def render_game_state(game_state):
    cell_images = create_cell_images(game_state)
    add_player_info(game_state, cell_images)
    add_grid(cell_images)
    return combine_cells_to_single_image(cell_images)


def create_cell_images(game_state, img_size=128):
    cell_images = []
    for y in range(game_state.map_height):
        row = []
        for x in range(game_state.map_width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                row.append(icons[cell.resource.type])
                #cell.resource.amount
            else:
                row.append(np.zeros((img_size, img_size, 4)))
        cell_images.append(row)
    return cell_images


def add_player_info(game_state, cell_images):
    for player_idx, player in enumerate(game_state.players):
        for city in player.cities.values():
            for city_tile in city.citytiles:
                img_base = cell_images[city_tile.pos.y][city_tile.pos.x]
                cell_images[city_tile.pos.y][city_tile.pos.x] = stack_images(img_base, apply_player_color(icons['city'], player_idx))

        for unit in player.units:
            img_base = cell_images[unit.pos.y][unit.pos.x]
            cell_images[unit.pos.y][unit.pos.x] = stack_images(img_base, apply_player_color(icons[unit_number_to_name[unit.type]], player_idx))


def stack_images(bottom, top):
    img = bottom.copy()
    mask = top[:, :, 3]
    for idx in range(4):
        img[:, :, idx] = img[:, :, idx]*(1.-mask) + mask*top[:, :, idx]
    return img


def apply_player_color(icon, player_idx):
    if player_idx:
        return icon[:, :, [2, 1, 0, 3]]
    else:
        return icon


def add_grid(cell_images, thickness=2, grid_color=(0, 0, 0, 1)):
    for row in cell_images:
        for img in row:
            img[:thickness] = grid_color
            img[:, :thickness] = grid_color
            img[-thickness:] = grid_color
            img[:, -thickness:] = grid_color


def combine_cells_to_single_image(cell_images):
    rows = [np.hstack(row) for row in cell_images]
    return np.vstack(rows)