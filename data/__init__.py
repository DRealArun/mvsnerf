from .llff import LLFFDataset
from .blender import BlenderDataset
from .dtu_ft import DTU_ft
from .dtu import MVSDatasetDTU
from .blender_temporal import BlenderTemporalDataset

dataset_dict = {'dtu': MVSDatasetDTU,
                'llff':LLFFDataset,
                'blender': BlenderDataset,
                'blender_temporal': BlenderTemporalDataset,
                'dtu_ft': DTU_ft}