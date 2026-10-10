from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset_ego_preserving import \
    EarlyFusionDatasetEgoPreserving
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset_warp import \
    IntermediateFusionDatasetWarp

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'EarlyFusionDatasetEgoPreserving': EarlyFusionDatasetEgoPreserving,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'IntermediateFusionDatasetWarp': IntermediateFusionDatasetWarp
}

GT_RANGE = [-100, -40, -5, 100, 40, 3]
COM_RANGE = 70

def build_dataset(dataset_cfg, visualize=False, train=True, isSim=False):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset','EarlyFusionDatasetEgoPreserving',
                            'IntermediateFusionDataset'], error_message

    fusion_args = dataset_cfg['fusion']['args']
    use_warp_pairwise = (isinstance(fusion_args, dict) and
                         fusion_args.get('warp_pairwise', False))
    if dataset_name == 'IntermediateFusionDataset' and use_warp_pairwise:
        dataset_name = 'IntermediateFusionDatasetWarp'

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        isSim=isSim
    )

    return dataset
