import os
import sys
import yaml
import shutil
import argparse
import numpy as np
from glob import glob


# Interpreter that has the FRNet stack (mmengine/mmcv/mmdet/mmdet3d/torch-scatter).
# Override with --frnet-python or the FRNET_PYTHON environment variable.
DEFAULT_FRNET_PYTHON = os.environ.get(
    "FRNET_PYTHON",
    "/home/ustc_user1/miniconda3/envs/frnet/bin/python",
)


def args_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument('--frnet-python', type=str, default=DEFAULT_FRNET_PYTHON,
                        help='Python interpreter that has the FRNet mmdet3d stack '
                             '(default: $FRNET_PYTHON or %s)' % DEFAULT_FRNET_PYTHON)
    args = parser.parse_args()
    return args


def read_ori_pcd(input_path):
    lidar = []
    with open(input_path, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)




def convert2bin(input_pcd_dir, output_bin_dir):
    file_list = sorted(glob(input_pcd_dir))
    if not os.path.exists(output_bin_dir):
        os.makedirs(output_bin_dir)
    for i, file in enumerate(file_list):
        p_xyzi = read_ori_pcd(file)
        p_xyzi = p_xyzi.reshape((-1, 4)).astype(np.float32)
        min_val = np.amin(p_xyzi[:, 3])
        max_val = np.amax(p_xyzi[:, 3])
        p_xyzi[:, 3] = (p_xyzi[:, 3] - min_val)/(max_val-min_val)
        p_xyzi[:, 3] = np.round(p_xyzi[:, 3], decimals=2)
        p_xyzi[:, 3] = np.minimum(p_xyzi[:, 3], 0.99)
        velodyne_file_new = os.path.join(output_bin_dir, f"{i + 1:06d}") + '.bin'
        p_xyzi.tofile(velodyne_file_new)


def copy_files(input_dir, output_dir, file_format="pcd"):
    file_list = sorted(glob(input_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, source_file in enumerate(file_list):
        des_file = os.path.join(output_dir, f"{i + 1:06d}.") + file_format
        shutil.copy(source_file, des_file)


if __name__ == '__main__':
    opt = args_parser()
    test_dir = opt.dataset_dir
    dataset_root = os.path.dirname(test_dir)

    v2x_dataset_dir = os.path.join(dataset_root, "v2x_dataset")

    if not os.path.exists(v2x_dataset_dir):
        os.makedirs(v2x_dataset_dir)

    for file_format in ["pcd", "yaml"]:
        ego_files = os.path.join(test_dir, f"0/*.{file_format}")
        cp_files = os.path.join(test_dir, f"1/*.{file_format}")

        des_ego_dir = os.path.join(v2x_dataset_dir, f"0/{file_format}")
        des_cp_dir = os.path.join(v2x_dataset_dir, f"1/{file_format}")

        copy_files(ego_files, des_ego_dir, file_format)
        copy_files(cp_files, des_cp_dir, file_format)

        if file_format == "pcd":
            ego_bin_dir = os.path.join(v2x_dataset_dir, f"0/velodyne")
            cp_bin_dir = os.path.join(v2x_dataset_dir, f"1/velodyne")
            convert2bin(ego_files, ego_bin_dir)
            convert2bin(cp_files, cp_bin_dir)

    ego_bin_dir = os.path.join(v2x_dataset_dir, f"0/velodyne/*.bin")
    cp_bin_dir = os.path.join(v2x_dataset_dir, f"1/velodyne/*.bin")

    semantic_dir = os.path.join(dataset_root, "semantic")

    des_bin_ego_dir = os.path.join(semantic_dir, "semanticKitti/sequences/11/velodyne")
    des_bin_cp_dir = os.path.join(semantic_dir, "semanticKitti/sequences/12/velodyne")

    copy_files(ego_bin_dir, des_bin_ego_dir, "bin")
    copy_files(cp_bin_dir, des_bin_cp_dir, "bin")

    # Per-point road/semantic segmentation with FRNet (third/FRNet).
    # The driver writes <result>/sequences/{11,12}/predictions/*.label, the same
    # layout the previous SalsaNext step produced, so the copy step below is
    # unchanged. It must run under the FRNet interpreter, not this one.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    frnet_repo = os.path.join(repo_root, "third", "FRNet")
    frnet_driver = os.path.join(repo_root, "tools", "frnet_infer_labels.py")

    if not os.path.isfile(frnet_driver):
        raise SystemExit(f"FRNet inference driver not found: {frnet_driver}")

    cmd = " ".join([
        f"'{opt.frnet_python}'",
        f"'{frnet_driver}'",
        f"--data-root '{semantic_dir}/semanticKitti'",
        "--sequences 11,12",
        f"--config '{frnet_repo}/configs/frnet/frnet-semantickitti_seg.py'",
        f"--checkpoint '{frnet_repo}/checkpoints/frnet-semantickitti_seg.pth'",
        f"--out-root '{semantic_dir}/result'",
    ])
    if os.system(cmd) != 0:
        raise SystemExit("FRNet inference failed; see output above")

    ego_prediction_results = os.path.join(semantic_dir, 'result/sequences/11/predictions/*.label')
    cp_prediction_results = os.path.join(semantic_dir, 'result/sequences/12/predictions/*.label')

    des_prediction_ego_dir = os.path.join(v2x_dataset_dir, f"0/predictions")
    des_prediction_cp_dir = os.path.join(v2x_dataset_dir, f"1/predictions")

    copy_files(ego_prediction_results, des_prediction_ego_dir, "label")
    copy_files(cp_prediction_results, des_prediction_cp_dir, "label")

    dataset_config = {"dataset_path": dataset_root}

    with open("./config/dataset_config.yml", "w") as config_file:
        yaml.dump(dataset_config, config_file, default_flow_style=False)

