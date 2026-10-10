import yaml

noise_variance = 0.03
noise_average = 0
noise_severity = 0.1
loss_rate = 0.1

lidar_position = (0, 0, 0)
extend_range = 2

horizontal_resolution = 0.199
vertical_resolution = 1.25
horizontal_left = 180
horizontal_right = -180

vertical_down = 15
vertical_up = -25

laser1_degree = -0.667

range = 200

r = 1

height_threshold = 1

initial_box_color = [1, 1, 1]
window_width = 1000
window_height = 800
window_name = "Pcd_Show"
render_point_size = 1.5
render_background_color = [0, 0, 0]
render_show_coordinate_frame = True

scene_latitude_dict = {
    0: (-25, 1.4),
    1: (-1, -4.2),
    2: (-1.667, 1.4),
    3: (-15.639, -1.4),
    4: (-11.31, 1.4),
    5: (0, -1.4),
    6: (-0.667, 4.2),
    7: (-8.843, -1.4),
    8: (-7.254, 1.4),
    9: (0.333, -4.2),
    10: (-0.333, 1.4),
    11: (-6.148, -1.4),
    12: (-5.333, 4.2),
    13: (1.333, -1.4),
    14: (0.667, 4.2),
    15: (-4, -1.4),
    16: (-4.667, 1.4),
    17: (1.667, -4.2),
    18: (1, 1.4),
    19: (-3.667, -4.2),
    20: (-3.333, 4.2),
    21: (3.333, -1.4),
    22: (2.333, 1.4),
    23: (-2.667, -1.4),
    24: (-3, 1.4),
    25: (7, -1.4),
    26: (4.667, 1.4),
    27: (-2.333, -4.2),
    28: (-2, 4.2),
    29: (15, -1.4),
    30: (10.333, 1.4),
    31: (-1.333, -1.4),
}

v2v_config = {
    'evaluate_angle': {
        1: -25,
        2: -15.639,
        3: -11.31,
        4: -8.843,
        5: -7.254,
        6: -6.148,
        7: -5.333,
        8: -4.667,
        9: -4,
        10: -3.667,
        11: -3.333,
        12: -3,
        13: -2.667,
        14: -2.333,
        15: -2,
        16: -1.667,
        17: -1.333,
        18: -1,
        19: -0.667,
        20: -0.333,
        21: 0,
        22: 0.333,
        23: 0.667,
        24: 1,
        25: 1.333,
        26: 1.667,
        27: 2.333,
        28: 3.333,
        29: 4.667,
        30: 7,
        31: 10.333,
        32: 15
    },
    "horizontal_resolution": 0.199,
    "vertical_down": 15,
    "vertical_up": -25,
    "noise": {
        "noise_variance": 0.03,
        "noise_average": 0,
        "noise_severity": 0.1,
        "loss_rate": 0.1,
    },
    "lidar_position": (0, 0, 0),
    "extend_range": 2,
    "horizontal_left": 180,
    "horizontal_right": -180,
    "range": 200,
    "r": 1,
    "height_threshold": 1
}

if __name__ == '__main__':
    with open('lidar_config.yml', 'w') as file:
        yaml.dump(v2v_config, file, default_style=False, allow_unicode=True)
