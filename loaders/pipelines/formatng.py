import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D
from mmdet3d.core.points import BasePoints
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.core.bbox import BaseInstance3DBoxes

@PIPELINES.register_module()
class RaCFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(RaCFormatBundle3D, self).__init__(class_names, with_gt=with_gt, with_label=with_label)

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)
            
        if 'radar_points' in results:
            if isinstance(results['radar_points'], list):
                radar_list = []
                for i in range(len(results['radar_points'])):
                    radar_points = results['radar_points'][i]
                    assert isinstance(radar_points, BasePoints)
                    radar_list.append(DC(radar_points.tensor))
                results['radar_points'] = radar_list
            else:
                assert isinstance(results['radar_points'], BasePoints)
                results['radar_points'] = DC(results['radar_points'].tensor)             

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ], dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        for key in [
                'gt_labels_static3d', 'gt_labels_dynamic3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_bboxes_static3d' in results:
            if isinstance(results['gt_bboxes_static3d'], BaseInstance3DBoxes):
                results['gt_bboxes_static3d'] = DC(
                    results['gt_bboxes_static3d'], cpu_only=True)
            else:
                results['gt_bboxes_static3d'] = DC(
                    to_tensor(results['gt_bboxes_static3d']))
        if 'gt_bboxes_dynamic3d' in results:
            if isinstance(results['gt_bboxes_dynamic3d'], BaseInstance3DBoxes):
                results['gt_bboxes_dynamic3d'] = DC(
                    results['gt_bboxes_dynamic3d'], cpu_only=True)
            else:
                results['gt_bboxes_dynamic3d'] = DC(
                    to_tensor(results['gt_bboxes_dynamic3d']))
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
