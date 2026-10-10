import math
import open3d as o3d
import numpy as np
from utils.v2X_file import load_yaml
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian, Latitude

lidar_config = {}
simulation_mode = "road"
vertical_resolution = 1.25


def lidar_simulation(mesh_obj, sim_mode='road'):
    global lidar_config, simulation_mode
    lidar_config = load_yaml('config/lidar_config.yml')
    simulation_mode = sim_mode

    rays_args = get_ray_args()

    rays_args = get_min_ray_args4render_by_obj(mesh_obj, rays_args)

    render_args = [
        lidar_config['noise']['noise_average'],
        lidar_config['noise']['noise_variance'],
        lidar_config['noise']['noise_severity'],
        lidar_config['noise']['loss_rate']
    ]

    pcd_obj = get_obj_pcd(mesh_obj, rays_args, render_args)

    return pcd_obj


def lidar_intensity_convert(bg_pcd, obj_pcd):
    avg_colors = np.mean(np.asarray(bg_pcd.colors), axis=0)
    obj_colors = np.tile(avg_colors, (np.asarray(obj_pcd.points).shape[0], 1))
    obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)

    return obj_pcd


def get_ray_args():
    horizontal_resolution = lidar_config['horizontal_resolution']
    horizontal_left = lidar_config['horizontal_left']
    horizontal_right = lidar_config['horizontal_right']
    vertical_down = lidar_config['vertical_down']
    vertical_up = lidar_config['vertical_up']
    return [horizontal_left, horizontal_right, vertical_down, vertical_up, horizontal_resolution]


def get_min_ray_args4render_by_obj(obj, rays_args):
    extend_range = lidar_config['extend_range']
    horizontal_left, horizontal_right = rays_args[0], rays_args[1]
    vertical_down, vertical_up = rays_args[2], rays_args[3]
    horizontal_resolution = rays_args[4]
    lidar_position = lidar_config['lidar_position']

    box_points = obj.get_oriented_bounding_box().get_box_points()

    temp = np.asarray(box_points)
    for point in temp:
        point -= lidar_position
        _, latitude, longitude = cartesian_to_spherical(*list(point))

        latitude, longitude = latitude.value, longitude.value
        if longitude > np.pi:
            longitude = -(np.pi * 2 - longitude)
        latitude = math.degrees(latitude)
        longitude = math.degrees(longitude)

        if horizontal_left > longitude:
            horizontal_left = longitude
        if horizontal_right < longitude:
            horizontal_right = longitude
        if vertical_down > latitude:
            vertical_down = latitude
        if vertical_up < latitude:
            vertical_up = latitude

    horizontal_left -= extend_range
    horizontal_right += extend_range
    vertical_down -= extend_range
    vertical_up += extend_range

    return [horizontal_left, horizontal_right, vertical_down, vertical_up, horizontal_resolution]


def get_obj_pcd(car, rays_args, render_args):
    car_t = o3d.t.geometry.TriangleMesh.from_legacy(car)
    car.paint_uniform_color([1, 1, 0])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(car_t)

    rays = get_rays(*rays_args)
    lidar_position = lidar_config['lidar_position']

    ans = scene.cast_rays(rays)
    distance = ans["t_hit"].numpy()

    xyz_direction = rays.numpy()[:, 3:]
    xyz_position = []

    for i in range(len(xyz_direction)):
        r, phi, theta = cartesian_to_spherical(*list(xyz_direction[i, :]))
        if distance[i] == np.inf:
            pass
        else:
            x, y, z = spherical_to_cartesian(distance[i], phi, theta)
            xyz_position.append([x, y, z + lidar_position[2]])
    points_obj = render_pcd(np.array(xyz_position), *render_args)

    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(points_obj)

    return pcd_obj


def get_rays(horizontal_left,
             horizontal_right,
             vertical_down,
             vertical_up,
             horizontal_resolution):
    points = ray_direction(horizontal_left,
                           horizontal_right,
                           vertical_down,
                           vertical_up,
                           horizontal_resolution,
                           lidar_config['r'])

    rays = create_rays(lidar_config['lidar_position'], points)
    return rays


def create_rays(lidar_position, point_directions):
    assert len(lidar_position) == 3
    rays = []
    for point_direction in point_directions:
        ray = (lidar_position[0], lidar_position[1], lidar_position[2],
               point_direction[0], point_direction[1], point_direction[2])
        rays.append(ray)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    return rays


def ray_direction(horizontal_left,
                  horizontal_right,
                  vertical_down,
                  vertical_up,
                  horizontal_resolution,
                  r):
    points_list = []
    if simulation_mode == 'road':
        lidar_dict = lidar_config['evaluate_angle']
        latitude_down = vertical_down * np.pi / 180
        latitude_up = vertical_up * np.pi / 180
        for idx, degree in lidar_dict.items():
            latitude = degree * np.pi / 180
            if latitude < latitude_down or latitude > latitude_up:
                continue
            pts = ray_direction_circle_simulation(horizontal_left, horizontal_right, horizontal_resolution, r, latitude)
            points_list += pts
    elif simulation_mode == 'vehicle':
        circle_num = int((vertical_up - vertical_down) // vertical_resolution)
        for i in range(circle_num):
            degree = vertical_resolution * i + vertical_down
            rad_phi = degree * np.pi / 180
            pts = ray_direction_circle_simulation(horizontal_left, horizontal_right, horizontal_resolution, r, rad_phi)
            points_list += pts

    return points_list


def ray_direction_circle_simulation(horizontal_left,
                                    horizontal_right,
                                    horizontal_resolution,
                                    r,
                                    rad_phi):
    laster_num = int((horizontal_right - horizontal_left) // horizontal_resolution)

    points_list = []

    for i in range(laster_num):
        degree = horizontal_resolution * i + horizontal_left
        rad_theta = degree * np.pi / 180
        x, y, z = spherical_to_cartesian(r, rad_phi, rad_theta)
        points_list.append((x, y, z))

    return points_list


def render_pcd(pointcloud_xyz, average, variance, severity, loss_rate):
    try:
        row, column = pointcloud_xyz.shape
    except ValueError as e:
        raise ValueError()
    jitter = np.random.normal(average, variance, size=(row, column)) * severity
    new_pc_xyz = (pointcloud_xyz + jitter).astype('float32')
    index = np.random.choice(row, size=int(row * (1 - loss_rate)), replace=False)
    return new_pc_xyz[index]

