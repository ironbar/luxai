"""
Utils for rendering games

TODO:
- [x] Improve background for opencv
- [x] Add information about resources
- [x] Add information about cooldown
- [x] Add caption information
- [x] Day and night
- [ ] Refactor
"""
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


img_paths = glob.glob('/home/gbarbadillo/Desktop/luxai_icons/128px/*.png')
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
            if cell.has_resource():
                cell_img = icons[cell.resource.type].copy()
                draw_text(cell_img, str(cell.resource.amount), position=(5, 30))
                row.append(stack_images(emtpy_cell.copy(), cell_img))
            else:
                row.append(emtpy_cell.copy())
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
                if city_tile.cooldown:
                    draw_text(img, str(int(city_tile.cooldown)), position=(5, 60))
                cell_images[city_tile.pos.y][city_tile.pos.x] = img

        for unit in player.units:
            img_base = cell_images[unit.pos.y][unit.pos.x]
            img = stack_images(img_base, apply_player_color(icons[unit_number_to_name[unit.type]], player_idx))
            # TODO: do not use hardcoded constant
            cargo = 100 - unit.get_cargo_space_left()
            if cargo:
                draw_text(img, str(cargo), position=(img.shape[1]-55, img.shape[0]-15))
            if unit.cooldown:
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
        return icon
    else:
        return icon[:, :, [2, 1, 0, 3]]


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


def get_citytile_pos_from_action(action):
    return [int(x) for x in action.split(' ')[1:]]
