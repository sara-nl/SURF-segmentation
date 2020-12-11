from xmlpathology.batchgenerator.data.dataset import DataSet
from ..utils import one_hot_encoding, fit_data
import numpy as np
from typing import Dict, Tuple


class SampleCallback:

    def __init__(self):
        pass

    def __call__(self,
                 dataset: DataSet,
                 x_patch: np.ndarray,
                 y_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        sample_auxiliaries_callback = {}
        return x_patch, y_patch, sample_auxiliaries_callback


class BatchCallback:

    def __init__(self):
        pass

    def __call__(self,
                 dataset: DataSet,
                 x_batch: np.ndarray,
                 y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        batch_auxiliaries_callback = {}
        return x_batch, y_batch, batch_auxiliaries_callback


class OneHotEncoding(SampleCallback):

    def __call__(self, dataset, x_patch, y_patch):
        y_patch = one_hot_encoding(y_patch, dataset.label_map)
        return x_patch, y_patch, {}


class FitData(SampleCallback):

    def __init__(self, output_shape):
        self._output_shape = output_shape

    def __call__(self, dataset, x_patch, y_patch):
        y_patch = self._fit_data(y_patch)
        return x_patch, y_patch, {}

    def _fit_data(self, y_patch):
        # cropping
        if y_patch.shape != self._output_shape:
            y_patch = fit_data(y_patch, self._output_shape)
        # Reshape
        return y_patch


class PixelCounter(BatchCallback):
    """
    Shpuld be applied after one hot encoding
    """

    def __init__(self):
        super().__init__()

    def __call__(self,
                 dataset: DataSet,
                 x_batch: np.ndarray,
                 y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:

        batch_auxiliaries_callback = self._one_hot_encoded_count(y_batch, dataset.label_map)
        return x_batch, y_batch, batch_auxiliaries_callback

    def _one_hot_encoded_count(self, y_batch: np.ndarray, label_map: Dict) -> Dict[str, Dict[int, int]]:
        inv_label_map_indexed = {label_index: label for label_index, label in enumerate(label_map)}
        count_per_label = np.sum(y_batch, axis=tuple(range(len(y_batch.shape)-1)))
        return {'pixel_count': {inv_label_map_indexed[label_index]: count for label_index, count in enumerate(count_per_label)}}


#TODO
class DataAugmentation(BatchCallback):
    def __init__(self, data_augmentation_config):
        self._data_augmentation_config = data_augmentation_config

    def __call__(self,
                 dataset: DataSet,
                 x_batch: np.ndarray,
                 y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:

                 return x_batch, y_batch, {}

# class MaskWeights:
#     def __init__(self, weight_type, output_shape):
#         self._weight_type = weight_type
#         self._output_shape = output_shape
#         self._mask_weights = None

#     def __call__(self, patch, mask):
#         if self._weight_type == 'batch':
#             self._mask_weights = self._batch_weights(mask)
#         else:
#             self._mask_weights = self._clean_weigths(mask)

#         # reshape
#         self._mask_weigths = self._mask_weights.reshape(self._output_shape[0] * self._output_shape[1])

#         return patch, mask, self._mask_weights

#     def _batch_weights(self, masks):

#         n_classes = masks.shape[-1]
#         # number of counts for each class
#         counts = np.zeros(n_classes)

#         # placeholder for weights
#         weights = np.zeros(masks.shape)

#         # for every example in batch count for each label pixels
#         for batch_index in range(len(masks)):
#             counts += [np.sum(masks[batch_index].reshape(-1, n_classes)[..., c_]) for c_ in range(n_classes)]

#         # multiply valid labels (1.0) * 1.0/counts for each label
#         for idx in range(n_classes):
#             if counts[idx] > 0:
#                 weights[..., idx] = masks[..., idx] * 1.0/counts[idx]

#         weights *= (np.sum(masks) / np.sum(weights))
#         return np.sum(weights, axis=-1)

#     def _clean_weights(self, masks):
#         return np.clip(np.sum(masks, axis=-1), 0, 1)


# def clean_weights(masks):
#     return np.clip(np.sum(masks, axis=-1), 0, 1)


# class FitYolo():

#     def convert_mask_to_yollo_output(mask, grid_shape, number_of_anchors, number_of_classes, output_shape, bounding_box_size):
#         yollo_label = np.zeros((*grid_shape, number_of_anchors, 4 + 1 + number_of_classes))

#         classes = list(np.unique(mask))
#         classes.remove(0)

#         for _class in classes:
#             transformed_points = np.where(mask == _class)
#             new_points = list(zip(transformed_points[0], transformed_points[1]))

#             for point in new_points:
#                 center_x = point[1] / (output_shape[0] / grid_shape[0])
#                 center_y = point[0] / (output_shape[1] / grid_shape[1])

#                 # skip points that are on the border and will be assigned to grid_cell out of range
#                 if center_x >= grid_shape[0] or center_y >= grid_shape[1]:
#                     continue

#                 grid_x = int(np.floor(center_x))
#                 grid_y = int(np.floor(center_y))

#                 # relative to grid cell
#                 center_w = bounding_box_size / (output_shape[0] / grid_shape[0])
#                 center_h = bounding_box_size / (output_shape[1] / grid_shape[1])
#                 box = [center_x, center_y, center_w, center_h]

#                 # TODO find best anchor
#                 anchor = 0

#                 # ground truth
#                 yollo_label[grid_y, grid_x, anchor, 0:4] = box
#                 yollo_label[grid_y, grid_x, anchor, 4] = 1.  # confidence
#                 yollo_label[grid_y, grid_x, anchor, 4 + int(_class)] = 1  # class

#         return yollo_label

#     def __init__(self,
#                  label_map,
#                  output_shape,
#                  grid_shape,
#                  number_of_anchors,
#                  bounding_box_size):

#         self._output_shape = output_shape
#         self._grid_shape = grid_shape
#         self._number_of_anchors = number_of_anchors
#         self._bounding_box_size = bounding_box_size
#         self._number_of_classes = len(label_map)

#     def __call__(self, dataset, patch, mask, weights):

#         yollo_label = FitYolo.convert_mask_to_yollo_output(mask,
#                                                            self._grid_shape,
#                                                            self._number_of_anchors,
#                                                            self._number_of_anchors,
#                                                            self._output_shape,
#                                                            self._bounding_box_size)
#         return patch, yollo_label, weights