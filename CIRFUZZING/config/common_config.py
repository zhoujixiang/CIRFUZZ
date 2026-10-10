import os
occlusion_th = 0.95
occ_point_max = 20

project_dir = "./"

assets_dir = "{}/_assets".format(project_dir)
obj_dir_path = "{}/shapenet".format(assets_dir)
obj_cp_dir = "{}/copy_paste".format(assets_dir)

obj_filename = "models/model_normalized.gltf"
multi_scale = 5.5
