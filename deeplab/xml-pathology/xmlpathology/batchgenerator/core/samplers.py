import abc

import cv2
import numpy as np
from shapely import geometry

from ..data.wholeslideimage import WholeSlideImageOpenSlide
from ..utils import shift_coordinates
from ..utils import log
import datetime
import pdb

def load_label_sampler_loader(label_sampler_type='segmentation'):
    if label_sampler_type == 'segmentation':
        return LabelSamplerLoader(class_=SegmentationLabelSampler)
    elif label_sampler_type == 'classification':
        return LabelSamplerLoader(class_=ClassificationLabelSampler)
    return None


class PatchSamplerLoader:

    def __init__(self, class_, **kwargs):
        self._class = class_
        self._kwargs = kwargs

    def __call__(self, dataset):
        return self._class(dataset=dataset, **self._kwargs)


class LabelSamplerLoader:

    def __init__(self, class_, **kwargs):
        self._class = class_
        self._kwargs = kwargs

    def __call__(self):
        return self._class(**self._kwargs)


class PointSamplerLoader:

    def __init__(self, class_, **kwargs):
        self._class = class_
        self._kwargs = kwargs

    def __call__(self, seed=123):
        return self._class(seed=seed, **self._kwargs)


class SamplerLoader:

    def __init__(self, class_, **kwargs):
        self._class = class_
        self._kwargs = kwargs

    def __call__(self,
                 dataset,
                 patch_sampler_loader,
                 label_sampler_loader,
                 point_sampler_loader,
                 sample_callbacks,
                 seed=123,
                 logging=log.Logger()):

        return self._class(dataset=dataset,
                           patch_sampler_loader=patch_sampler_loader,
                           label_sampler_loader=label_sampler_loader,
                           point_sampler_loader=point_sampler_loader,
                           sample_callbacks=sample_callbacks,
                           seed=seed,
                           logging=logging, **self._kwargs)


class BatchSamplerLoader:

    def __init__(self, class_, **kwargs):
        self._class = class_
        self._kwargs = kwargs

    def __call__(self,
                 dataset,
                 sampler_loader,
                 patch_sampler_loader,
                 label_sampler_loader,
                 point_sampler_loader,
                 batch_callbacks,
                 sample_callbacks,
                 seed=123,
                 logging=log.Logger()):

        return self._class(
            dataset=dataset,
            sampler_loader=sampler_loader,
            patch_sampler_loader=patch_sampler_loader,
            label_sampler_loader=label_sampler_loader,
            point_sampler_loader=point_sampler_loader,
            batch_callbacks=batch_callbacks,
            sample_callbacks=sample_callbacks,
            seed=seed,
            logging=logging, **self._kwargs)


class PatchSampler:

    def __init__(self, dataset):
        self._dataset = dataset

    def sample(self, image_path, center_x, center_y, width, height, pixel_spacing):
        if self._dataset.images:
            image = self._dataset.images[image_path]
        else:
            image = WholeSlideImageOpenSlide(image_path=image_path)
        patch = np.array(image.get_patch(int(center_x), int(center_y), int(width), int(height), float(pixel_spacing)))
        if not self._dataset.images:
            image.close()
            image = None
        return patch.astype('uint8')


class LabelSampler:

    def __init__(self):
        pass

    @abc.abstractmethod
    def sample(self, image_annotation, annotation, center_x, center_y, width, height, pixel_spacing):
        pass


class SegmentationLabelSampler(LabelSampler):

    def __init__(self):
        super().__init__()

    def sample(self, image_annotation, annotation, center_x, center_y, width, height, pixel_spacing):
        # get ratio
        ratio = image_annotation.get_ratio(pixel_spacing)
        # get annotations
        annotations = image_annotation.select_annotations(center_x, center_y, (width*ratio)-1, (height*ratio)-1)
        
        # create mask placeholder
        mask = np.zeros((height, width), dtype=np.int32)
        # set labels of all selected annotations
        for annotation in annotations:
            coordinates = annotation.coordinates()
            coordinates = shift_coordinates(coordinates, center_x, center_y, width, height, ratio)

            if annotation.type == 'polygon':
                holemask = np.ones((width, height), dtype=np.int32)*-1
                for hole in annotation.holes():
                    hcoordinates = shift_coordinates(hole, center_x, center_y, width, height, ratio)
                    cv2.fillPoly(holemask, np.array([hcoordinates], dtype=np.int32), 1)
                    holemask[holemask != -1] = mask[holemask != -1]
                cv2.fillPoly(mask, np.array([coordinates], dtype=np.int32), annotation.label_value)
                mask[holemask != -1] = holemask[holemask != -1]

            elif annotation.type == 'dot':
                mask[int(coordinates[1]), int(coordinates[0])] = annotation.label_value

        return mask.astype('uint8')


class ClassificationLabelSampler(LabelSampler):
    def __init__(self):
        super().__init__()

    def sample(self, image_annotation, annotation, center_x, center_y, width, height, pixel_spacing):
        return annotation.label_value


# class DetectionLabelSampler(BaseSampler):
#     def __init__(self, sample_type):
#         super().__init__(sample_type)


class PointSampler():
    def __init__(self, seed=123):
        self._seed = seed
        np.random.seed(self._seed)

    def set_seed(self):
        np.random.seed(self._seed)

    @abc.abstractmethod
    def sample(self, polygon, width, height, ratio):
        pass


class CenterPointSampler(PointSampler):
    """ samples center point"""

    def __init__(self, seed=123):
        super().__init__(seed=seed)
        np.random.seed(seed)

    def sample(self, polygon, width, height, ratio):
        return polygon.center

    def reset(self):
        self.set_seed()


class RandomPointSampler(PointSampler):
    """ samples points randomly within a polygon with a max limit of 100 otherwise it will return the centroid """
    MAX_ITTERATION = 100

    def __init__(self, strict_point_sampling=False, seed=123):
        super().__init__(seed=seed)
        np.random.seed(seed)
        self._strict_point_sampling = strict_point_sampling

    def sample(self, polygon, width, height, ratio):
        for _ in range(RandomPointSampler.MAX_ITTERATION):
            x_min, y_min, x_max, y_max = polygon.bounds
            x_c, y_c = np.random.uniform(
                x_min, x_max), np.random.uniform(y_min, y_max)
            center_shape = geometry.box(x_c-(width*ratio)//2, y_c-(height*ratio)//2, x_c+(width*ratio)//2, y_c +
                                        (height*ratio)//2) if self._strict_point_sampling else geometry.Point(x_c, y_c)
            if polygon.buffer(0).contains(center_shape):
                return x_c, y_c

        return polygon.centroid

    def reset(self):
        self.set_seed()


# class DensityPointSampler(PointSampler):
#     """ samples points based on the density map of the points within a polygon """

#     def __init__(self, inverse_density_ratio=0.2, seed=123):
#         super().__init__(seed=seed)
#         self._inverse_density_ratio = inverse_density_ratio

#     def sample(self, polygon, width, height, ratio):
#         if self._inverse_density_ratio < np.random.rand():
#             choices = (np.random.rand(*polygon.density_map.shape)
#                        < polygon.density_map).astype(int)
#         else:
#             choices = (np.random.rand(*polygon.density_map.shape)
#                        < 1-polygon.density_map).astype(int)

#         choice_indexes = np.where(choices)
#         # if points in annotation
#         if choice_indexes[0].shape[0]:
#             index = np.random.randint(choice_indexes[0].shape[0])
#             x_index, y_index = choice_indexes[1][index], choice_indexes[0][index]

#             # offset the x and y coordinates of the sampled patch
#             x_c, y_c = x_index*polygon.density_downsampling_ratio, y_index * \
#                 polygon.density_downsampling_ratio

#             x_min, y_min, _, _ = polygon.bounds
#             x_c, y_c = x_c+x_min, y_c+y_min

#         # if no point in annotation
#         else:
#             x_min, y_min, x_max, y_max = polygon.bounds
#             x_c, y_c = np.random.uniform(
#                 x_min, x_max), np.random.uniform(y_min, y_max)

#         return x_c, y_c


class Sampler():
    def __init__(self,
                 dataset,
                 input_shapes=((256, 256, 3)),
                 spacings=(0.5),
                 patch_sampler_loader=PatchSamplerLoader(class_=PatchSampler),
                 label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                 point_sampler_loader=PointSamplerLoader(class_=RandomPointSampler, strict_point_sampling=False),
                 sample_callbacks=None,
                 seed=123,
                 logging=log.Logger()):

        np.random.seed(seed)

        # create logger
        # self._logger = logging.get_logger(f'Sampler-{dataset.mode}-{seed}')

        self._dataset = dataset
        self._mode = dataset.mode

        self._input_shapes = input_shapes
        self._spacings = spacings

        self._patch_sampler = patch_sampler_loader(dataset=dataset)
        self._label_sampler = label_sampler_loader()
        self._point_sampler = point_sampler_loader(seed=seed)

        self._sample_callbacks = sample_callbacks

        self._seed = seed

    def sample(self, image_annotation_index, annotation_index, i=None):

        x_samples = [[None for spacing in self._spacings] for shape in self._input_shapes]
        y_samples = [[None for spacing in self._spacings] for shape in self._input_shapes]
        sample_auxiliaries = {}

        # sample center_coordinates
        center_coordinates = self._sample_center_coordinates(image_annotation_index, annotation_index)

        sample_infos = []
        for shape_index, patch_shape in enumerate(self._input_shapes):
            for spacing_index, pixel_spacing in enumerate(self._spacings):

                # sample patch, mask
                patch, mask, sample_info = self._sample(image_annotation_index, annotation_index, center_coordinates, patch_shape, pixel_spacing)
                sample_info.update({'batch': i})
                sample_infos.append(sample_info)
                # apply sample callbacks
                patch, mask, auxiliaries = self._apply_sample_callbacks(patch, mask)
                sample_auxiliaries.update(auxiliaries)

                # add samples to list
                x_samples[shape_index][spacing_index] = patch
                y_samples[shape_index][spacing_index] = mask

        # add sample info to sample auxiliaries
        sample_auxiliaries[f'sample_info'] = sample_infos

        # resolve samples
        x_samples, y_samples = map(self._resolve_samples, [x_samples, y_samples])

        return x_samples, y_samples, sample_auxiliaries

    def _sample_center_coordinates(self, image_annotation_index, annotation_index):
        # get image annotation
        image_annotation = self._dataset.image_annotations[image_annotation_index]
        
        # get annotation
        annotation = image_annotation.annotations[annotation_index]

        # sample from data (largest patch_size lowest resolution for strict sampling)
        return self._point_sampler.sample(annotation,
                                          self._input_shapes[-1][0],
                                          self._input_shapes[-1][1],
                                          image_annotation.get_ratio(self._spacings[-1]))

    def _sample(self, image_annotation_index, annotation_index, center_coordinates, patch_shape, pixel_spacing):
        # get image path
        image_annotation = self._dataset.image_annotations[image_annotation_index]
        annotation = image_annotation.annotations[annotation_index]
        width, height, _ = patch_shape
        center_x, center_y = center_coordinates
        image_path = self._dataset.get_image_path(image_annotation.data_source_id)

        # sample patch
        patch = self._patch_sampler.sample(image_path, center_x, center_y, width, height, pixel_spacing)

        # sample label
        label = self._label_sampler.sample(image_annotation, annotation, center_x, center_y, width, height, pixel_spacing)

        # set sample info
        sample_info = {'label': annotation.label_name,
                       'image_annotation_index': image_annotation_index,
                       'annotation_index': annotation_index,
                       'image': image_annotation.image_path,
                       'mode': self._dataset.mode,
                       'center': list(map(int, center_coordinates)),
                       'pixel_spacing': pixel_spacing,
                       'patch_shape': patch_shape,
                       'datetime:': str(datetime.datetime.now())}

        return patch, label, sample_info

    def _apply_sample_callbacks(self, patch, mask):
        sample_auxiliaries = {}
        # apply callbacks
        if self._sample_callbacks:
            for callback in self._sample_callbacks:
                patch, mask, sample_auxiliaries_from_callback = callback(self._dataset, patch, mask)
                sample_auxiliaries.update(sample_auxiliaries_from_callback)
        return patch, mask, sample_auxiliaries

    def _resolve_samples(self, batch):
        axis_index = 0
        for _ in range(2):
            try:
                batch = np.array(batch).squeeze(axis=axis_index)
            except ValueError:
                axis_index = 1
        return np.array(batch)

    def reset(self):
        self._point_sampler.reset()


class WSISampler(Sampler):
    def __init__(self,
                 dataset,
                 input_shapes=((256, 256, 3)),
                 spacings=(0.5),
                 patch_sampler_loader=PatchSamplerLoader(class_=PatchSampler),
                 label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                 point_sampler_loader=PointSamplerLoader(class_=RandomPointSampler, strict_point_sampling=False),
                 sample_callbacks=None,
                 seed=123,
                 logging=log.Logger()):

        super().__init__(dataset,
                         input_shapes,
                         spacings,
                         patch_sampler_loader,
                         label_sampler_loader,
                         point_sampler_loader,
                         sample_callbacks,
                         seed,
                         logging)

    def _sample(self, image_annotation_index, annotation_index, center_coordinates, patch_shape, pixel_spacing):
        width, height = patch_shape
        center_x, center_y = center_coordinates
        image_annotation = self._dataset.image_annotations[image_annotation_index]
        annotation = image_annotation.annotations[annotation_index]
        mask_path = self._dataset.get_mask_path(image_annotation.data_source_id)
        if mask_path:
            mask_patch = self._patch_sampler.sample(mask_path, center_x, center_y, width, height, pixel_spacing)
            if not np.any(mask_patch):
                return np.zeros((width, height, 3)), None, None
        image_path = self._dataset.get_image_path(image_annotation.data_source_id)
        patch = self._patch_sampler.sample(image_path, center_x, center_y, width, height, pixel_spacing)
        return patch, None, {}


class BatchSampler():
    def __init__(self,
                 dataset,
                 sampler_loader=SamplerLoader(class_=Sampler, input_shapes=[(256, 256, 3)], spacings=[0.5]),
                 patch_sampler_loader=PatchSamplerLoader(class_=PatchSampler),
                 label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                 point_sampler_loader=PointSamplerLoader(class_=RandomPointSampler, strict_point_sampling=False),
                 batch_callbacks=None,
                 sample_callbacks=None,
                 seed=123,
                 logging=log.Logger()):

        self._dataset = dataset
        np.random.seed(seed)

        self._sampler = sampler_loader(dataset=dataset,
                                       patch_sampler_loader=patch_sampler_loader,
                                       label_sampler_loader=label_sampler_loader,
                                       point_sampler_loader=point_sampler_loader,
                                       sample_callbacks=sample_callbacks,
                                       seed=seed,
                                       logging=logging)

        self._batch_callbacks = batch_callbacks

    def batch(self, batch_data, i=None):
        x_batch = []
        y_batch = []
        auxiliaries = {}

        x_batch, y_batch, sample_auxiliaries = self._sample_batch(batch_data, i)
        auxiliaries.update(sample_auxiliaries)

        x_batch, y_batch, callback_auxiliaries = self._apply_batch_callbacks(x_batch, y_batch)
        auxiliaries.update(callback_auxiliaries)

        return {'x_batch': np.array(x_batch), 'y_batch': np.array(y_batch), 'auxiliaries': auxiliaries}

    def _sample_batch(self, batch_data, i):

        # decalare x_batch and y_batch
        x_batch = []
        y_batch = []

        # set auxilaries
        auxiliaries = {'sampler': []}

        for sample_data in batch_data:

            # unpack sample data
            image_annotation_index, annotation_index = sample_data

            # sample
            x_samples, y_samples, sample_auxiliaries = self._sampler.sample(
                image_annotation_index, annotation_index, i=i)

            # append sampler auxiliaries
            auxiliaries['sampler'].append(sample_auxiliaries)

            # append samples to batch
            x_batch.append(x_samples)
            y_batch.append(y_samples)

        return x_batch, y_batch, auxiliaries

    def _apply_batch_callbacks(self, x_batch, y_batch):
        # set auxilaries
        auxiliaries = {}

        if self._batch_callbacks:
            for callback in self._batch_callbacks:
                x_batch, y_batch, batch_auxiliaries_from_callback = callback(
                    self._dataset, x_batch, y_batch)
                auxiliaries.update(batch_auxiliaries_from_callback)
        return x_batch, y_batch, auxiliaries

    def reset(self):
        self._sampler.reset()
