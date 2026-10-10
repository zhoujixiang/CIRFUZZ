# road_prediction

Standalone project for **road prediction dataset initialization** — extracted from
`V2XGen` so the dataset pipeline can run independently of the fuzzing codebase.

## What it does

[dataset_init.py](dataset_init.py) takes a raw V2V4Real-style test dataset directory
(`-d`) and builds a semantic road-segmented dataset:

1. Copies each agent's PCD point clouds and YAML annotations into a `v2x_dataset/` tree.
2. Converts PCD → binary velodyne format (SemanticKITTI layout).
3. Runs the bundled **SalsaNext** semantic segmentation network to produce per-point
   road predictions (`predictions/*.label`).
4. Records the dataset root in `config/dataset_config.yml` for downstream scripts.

## Layout

```
road_prediction/
├── dataset_init.py          # entry point
├── third/SalsaNext/         # SalsaNext semantic segmentation (train/tasks/semantic)
├── config/
│   └── dataset_config.yml   # dataset root, written by dataset_init.py
├── requirements.txt
└── .gitattributes           # SalsaNext weights tracked with Git LFS
```

## Usage

```bash
# from the project root (paths inside dataset_init.py are relative to CWD)
python dataset_init.py -d /path/to/your/v2v4real/testset

# example (original):
# cd /bdata/usrdata/root/v2x/V2XGen && python dataset_init.py -d /bdata/usrdata/root/v2x/V2V4Real/v2v4real
```

The input `-d <test_dir>` is expected to contain two subdirs `0/` and `1/` holding
`*.pcd` and `*.yaml` files per agent.

**Multi-scene test data.** V2V4Real `test` data comes as several `testoutput_*`
scene dirs (each with its own `0/`, `1/`, frame numbers restarting at 0). Use
[merge_test_scenes.py](merge_test_scenes.py) to merge them into one dir that
satisfies `dataset_init.py`'s layout (frames re-numbered globally, `0/` and `1/`
share the same frame number per frame):

```bash
python merge_test_scenes.py -s /path/to/Data/test -o /path/to/merged_test
python dataset_init.py -d /path/to/merged_test
```

Outputs are written next to the input, in the **parent** directory:

- `<parent>/v2x_dataset/` — pcd / yaml / velodyne / predictions per agent (`0/`, `1/`)
- `<parent>/semantic/semanticKitti/sequences/{11,12}/velodyne` — bin files for SalsaNext
- `<parent>/semantic/result/sequences/{11,12}/predictions/*.label` — predictions
- `config/dataset_config.yml` — records `<parent>` as `dataset_path`

## Notes

- `dataset_init.py` runs `python infer.py` via `os.system` from inside
  `third/SalsaNext/train/tasks/semantic`, so **run it from the project root** and keep
  `third/SalsaNext`'s relative structure intact. Make sure `python` on `PATH` is a
  Python 3.7/3.8 env with `torch`, `scipy`, `opencv-python` (e.g. the repo's
  `v2v4real` conda env): `export PATH=/path/to/v2v4real/bin:$PATH`.
- The SalsaNext `pretrained/SalsaNext` weights (~51 MB) are Git-LFS tracked. Clone with
  `git lfs pull` or set up `.lfsconfig` before running inference.
- No GPU is required to build the dataset, but SalsaNext inference will use CUDA when
  available and otherwise falls back to CPU.
