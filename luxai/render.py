"""
Utils for rendering games

TODO:
- [x] Improve background for opencv
- [x] Add information about resources
- [x] Add information about cooldown
- [x] Add caption information
- [x] Day and night
- [x] Update cart icon, it would be better if it has a similar size to the worker
- [x] Move icons to the repo
- [ ] Refactor
"""
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_objects import Unit, CityTile

from luxai.primitives import get_unit_cargo, is_cart

img_paths = glob.glob('/mnt/hdd0/MEGA/AI/22 Kaggle/luxai/data/render_icons/*.png')
icons = {os.path.splitext(os.path.basename(img_path))[0]: plt.imread(img_path) for img_path in img_paths}

unit_number_to_name = {0: 'worker', 1: 'cart'}


def render_game_state(game_state):
    cell_images = create_cell_images(game_state)
    add_player_info(game_state, cell_images)
    add_grid(cell_images)
    render = combine_cells_to_single_image(cell_images)
    return render


def create_cell_images(game_state, img_size=128):
    cell_images = []
    emtpy_cell = _get_emtpy_cell(img_size, game_state)
    for y in range(game_state.map_height):
        row = []
        for x in range(game_state.map_width):
            cell = game_state.map.get_cell(x, y)
            cell_img = emtpy_cell.copy()
            if cell.has_resource():
                cell_img = stack_images(cell_img, icons[cell.resource.type].copy())
                draw_text(cell_img, str(cell.resource.amount), position=(5, 30))
            if cell.road:
                draw_text(cell_img, '%.1f' % cell.road, position=(5, 118))
            row.append(cell_img.copy())
        cell_images.append(row)
    return cell_images


def draw_text(img, text, position, color=(0, 0, 0, 1), border_color=(0.5, 0.5, 0.5, 1)):
    """ Modifies the input image by writing the desired text at position """
    # cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=1, color=border_color, thickness=5)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=color, thickness=5)


def _get_emtpy_cell(img_size, game_state):
    """ Creates a green empty cell """
    emtpy_cell = np.ones((img_size, img_size, 4))*0.75
    emtpy_cell[:, :, 3] = 1 # alpha channel
    if not is_night(game_state):
        emtpy_cell[:, :, 1] = 1 # green
    return emtpy_cell


def is_night(game_state):
    return game_state.turn % 40 >= 30


def add_player_info(game_state, cell_images):
    for player_idx, player in enumerate(game_state.players):
        for city in player.cities.values():
            turns_can_survive_at_night = int(city.fuel//city.get_light_upkeep())
            for city_tile in city.citytiles:
                img_base = cell_images[city_tile.pos.y][city_tile.pos.x]
                img = stack_images(img_base, apply_player_color(icons['city'], player_idx))
                draw_text(img, str(turns_can_survive_at_night), position=(5, 30))
                if city_tile.cooldown >= 1:
                    draw_text(img, str(int(city_tile.cooldown)), position=(5, 60))
                cell_images[city_tile.pos.y][city_tile.pos.x] = img

        for unit in player.units:
            img_base = cell_images[unit.pos.y][unit.pos.x]
            img = stack_images(img_base, apply_player_color(icons[unit_number_to_name[unit.type]], player_idx))
            cargo = get_unit_cargo(unit)
            if cargo:
                draw_text(img, str(cargo), position=(img.shape[1]-65, img.shape[0]-10))
            if unit.cooldown >= 1:
                if is_cart(unit):
                    draw_text(img, '%.1f' % unit.cooldown, position=(img.shape[1]-60, img.shape[0]-45))
                else:
                    draw_text(img, str(int(unit.cooldown)), position=(img.shape[1]-40, img.shape[0]-45))
            cell_images[unit.pos.y][unit.pos.x] = img


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


def get_captions(game_state):
    captions = ''
    for player_idx, player in enumerate(game_state.players):
        captions += 'Player %i. Research points: %i. Citytiles: %i. Units: %i ' % (
            player_idx,
            player.research_points,
            sum(len(city.citytiles) for city in player.cities.values()),
            len(player.units))
        if player.researched_uranium():
            captions += 'Uranium era'
        elif player.researched_coal():
            captions += 'Coal era'
        captions += '\n'
    return captions[:-1]


def add_actions_to_render(render, actions, game_state):
    render = render.copy()
    for action in actions:
        _add_action_to_render(render, action, game_state)
    return render


def _add_action_to_render(render, action, game_state):
    if any(action.startswith(start) for start in ['r ', 'bw ', 'bc ']):
        x, y = get_citytile_pos_from_action(action)
        draw_text(render, action.split(' ')[0].upper(), position=(x*128+5, y*128 + 60))
    else:
        x, y = get_unit_pos_from_action(action, game_state)
        if action.startswith('bcity '):
            draw_text(render, 'B', position=((x + 1)*128 -40, (y+1)*128 - 45))
        elif action.startswith('m '):
            direction = action.split(' ')[-1]
            if direction == 'c':
                return
            draw_movement_arrow(render, x, y, direction)
        elif action.startswith('t '):
            destination_x, destination_y = get_unit_pos_from_unit_id(action.split(' ')[2], game_state)
            if destination_x < x:
                direction = 'w'
            elif destination_x > x:
                direction = 'e'
            elif destination_y < y:
                direction = 'n'
            elif destination_y > y:
                direction = 's'
            draw_movement_arrow(render, x, y, direction, (0, 0, 1., 0.5))


def draw_movement_arrow(render, x, y, direction, color=(0, 0, 0, 0.5)):
    arrow_origin = ((x + 1)*128 -43, (y+1)*128 - 60)
    arrow_len = 20
    if direction == 'n':
        arrow_end = (arrow_origin[0], arrow_origin[1] - arrow_len)
    elif direction == 's':
        arrow_end = (arrow_origin[0], arrow_origin[1] + arrow_len)
    elif direction == 'e':
        arrow_end = (arrow_origin[0] + arrow_len, arrow_origin[1])
    elif direction == 'w':
        arrow_end = (arrow_origin[0] - arrow_len, arrow_origin[1])
    cv2.arrowedLine(render, arrow_origin, arrow_end, color, 5, tipLength=0.5)


def get_citytile_pos_from_action(action):
    return [int(x) for x in action.split(' ')[1:]]


def get_unit_pos_from_action(action, game_state):
    unit_id = action.split(' ')[1]
    return get_unit_pos_from_unit_id(unit_id, game_state)


def get_unit_pos_from_unit_id(unit_id, game_state):
    for unit in game_state.players[0].units:
        if unit.id == unit_id:
            return unit.pos.x, unit.pos.y


def show_focus_on_active_unit(render, unit, color=(0, 0, 0.5, 1)):
    """ Modifies the input render by adding a circle around the active unit """
    if isinstance(unit, CityTile):
        center = (int(unit.pos.x*128 + 64), int(unit.pos.y*128 + 64))
        radius = 96
    else:
        center = (int(unit.pos.x*128 + 85), int(unit.pos.y*128 + 96))
        radius = 64
    cv2.circle(render, center, radius, color=color, thickness=3)