'''
This file defines XmlBatchgenerators

'''

import multiprocessing
import os

import numpy as np

from .data import dataset
from .core import controllers
from .core import samplers
import numpy as np


from .data.wholeslideannotation import AnnotationParserLoader, AsapAnnotationParser, WSIAnnotationParser
from .data.dataset import DataSet, DataSetLoader
from .utils import log, normalize
from .core import managers
from .core.controllers import (BalancedIndexController, BalancedLabelController,
                               IndexControllerLoader, LabelControllerLoader, SamplerControllerLoader, SamplerController)
from .core.samplers import (BatchSampler, BatchSamplerLoader, LabelSamplerLoader,
                            PatchSampler, PatchSamplerLoader, PointSamplerLoader,
                            RandomPointSampler, Sampler, SamplerLoader,
                            SegmentationLabelSampler, load_label_sampler_loader)



# multiprocessing.set_start_method('spawn', force=True)
# multiprocessing.set_start_method('forkserver')


class BatchGenerator():
    """
    An xml batch generation class which uses loaders and classes that all can be customized by the user.

    Args:
        data_sources (dict):
            specification of datasource expecting the following hierachie (following is depedent by the annoation_parser_loader (see args)):
            {mode: [{image_path: full_path_to_image_fill,
                    annotation_path: full_path_to_annotation_file}]}. were mode can be training or validation
        label_map (dict): label mapping between label names and label values.
        batch_size: number of examples in batch
        data_set_loader: loader class for loading a dataset (training/validation)
        annototaion_parser_loader: loader class for parsing annotations
        sampler_loader: loader class for sampling patches and ground truth (and weights)
        batch_sampler_loader: loader class for sampling batches
        label_controller_loader: loader class for sampling a new label
        index_controller_loader: loader class for sampling a new annotation (given a label)
        label_sampler_loader: loader class for sampling ground truth
        patch_sampler_loader: loader class for sampling a patch
        point_sampler_loader: loader class for sampling a center point wihtin a given annotation
        seed: seed number for random choices
        queue_size: size of number of batches to maintain in memmory
        cpus: number of cpus: 1 or >4 for multiprocessing
        normalizer: normalize function for patches
        sample_callbacks: callbacks applied to patch, mask, (weights)
        batch_callbacks: callbacks applied to the batch

    """

    def __init__(self,
                 data_sources,
                 label_map,
                 batch_size=1,
                 data_set_loader: DataSetLoader = DataSetLoader(class_=DataSet),
                 annotation_parser_loader: AnnotationParserLoader = AnnotationParserLoader(class_=AsapAnnotationParser, sort_by='label_map'),
                 sampler_loader=SamplerLoader(class_=Sampler, input_shapes=[(256, 256, 3)], spacings=[0.5]),
                 batch_sampler_loader=BatchSamplerLoader(class_=BatchSampler),
                 sampler_controller_loader=SamplerControllerLoader(class_=SamplerController),
                 label_controller_loader=LabelControllerLoader(class_=BalancedLabelController),
                 index_controller_loader=IndexControllerLoader(class_=BalancedIndexController),
                 patch_sampler_loader=PatchSamplerLoader(class_=PatchSampler),
                 label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                 point_sampler_loader=PointSamplerLoader(class_=RandomPointSampler, strict_point_sampling=False),
                 seed=123,
                 queue_size=40,
                 cpus=1,
                 normalizer=normalize,
                 sample_callbacks=None,
                 batch_callbacks=None,
                 log_path=None):

        # load datasets
        self._datasets = {mode: data_set_loader(mode=mode,
                                                data_source=data_source,
                                                label_map=label_map,
                                                annotation_parser_loader=annotation_parser_loader
                                                ) for mode, data_source in data_sources.items()}

        # setup logger
        self._logging = log.Logger(self._datasets, log_path)
        self._logger = self._logging.get_logger('generator')

        self._batch_manager = managers._get_batch_manager(datasets=self._datasets,
                                                          batch_size=batch_size,
                                                          data_set_loader=data_set_loader,
                                                          annotation_parser_loader=annotation_parser_loader,
                                                          sampler_loader=sampler_loader,
                                                          batch_sampler_loader=batch_sampler_loader,
                                                          sampler_controller_loader=sampler_controller_loader,
                                                          label_controller_loader=label_controller_loader,
                                                          index_controller_loader=index_controller_loader,
                                                          patch_sampler_loader=patch_sampler_loader,
                                                          label_sampler_loader=label_sampler_loader,
                                                          point_sampler_loader=point_sampler_loader,
                                                          seed=seed,
                                                          queue_size=queue_size,
                                                          cpus=cpus,
                                                          normalizer=normalizer,
                                                          sample_callbacks=sample_callbacks,
                                                          batch_callbacks=batch_callbacks,
                                                          logging=self._logging)

    @property
    def datasets(self):
        return self._datasets

    def batch(self, mode):
        """returns a batch for a given mode (e.g.: training, validation)"""
        batch = self._batch_manager.batch(mode)
        self._logging.update(batch['auxiliaries'])
        return batch

    def start(self):
        """starts the batch generator (mandatory for multiprocessing (i.e., cpus>=4)"""
        self._batch_manager.start()

    def reset(self, mode):
        """reset the batch generator  (currently not working)"""
        self._batch_manager.reset(mode)
        self._logger.info(f'BatchGenerator resetted {mode}')

    # @property
    # def datasets(self):
    #     return self._datasets

    def stop(self):
        """stops the batch generator and merges all logparts"""
        self._logger.info(f'Stopping BatchGenerator')
        self._batch_manager.stop()
        self._logger.info(f'Finalizing logging...')
        self._logging.finalize()
        for h in self._logger.handlers:
            h.close()


class BatchGeneratorVanilla(BatchGenerator):

    """
    An XmlBatchGenerator that has an easy interface that allows for basic functionality of the XmlBatchGenerator

    Args:
        data_sources (dict):
            specification of datasource expecting the following hierachie (following is depedent by the annoation_parser_loader (see args)):
            {mode: {image_path: full_path_to_image_file,
                    annotation_path: full_path_to_annotation_file}}. Example modes: training, validation
        label_map (dict): label mapping between label names and label values.
        batch_size (int): number of examples in batch
        input_shape (tuple): size of the patches
        spacing (float): pixel spacing of patches
        task (str): segmentation/classification/(TODO: detection)
        seed (int): seed for randomness
        cpus (int): cpu cores that are used by the batch generator (>3 allows for multiprocessing)
        normalizer (func): normalizer to normalize patch data
        sample_callbacks (list): callbacks applied to patch, mask
        batch_callbacks (list): callbacks applied to the batch
    """

    def __init__(self,
                 data_sources,
                 label_map,
                 batch_size=1,
                 input_shape=(256, 256, 3),
                 spacing=0.5,
                 task='segmentation',
                 strict_point_sampling=False,
                 seed=123,
                 cpus=1,
                 normalizer=normalize,
                 open_images_ahead=False,
                 batch_callbacks=None,
                 sample_callbacks=None,
                 log_path=None):

        # create dataset loader
        data_set_loader = DataSetLoader(class_=DataSet, open_images_ahead=open_images_ahead)

        # create sampler loader to load sampler
        sampler_loader = SamplerLoader(class_=Sampler, input_shapes=[input_shape], spacings=[spacing])

        # create point sampler loader
        point_sampler_loader = PointSamplerLoader(class_=RandomPointSampler, strict_point_sampling=strict_point_sampling)

        # create label sampler loader for label sampling based on the given task
        label_sampler_loader = load_label_sampler_loader(task)

        super().__init__(data_sources=data_sources,
                         label_map=label_map,
                         batch_size=batch_size,
                         data_set_loader=data_set_loader,
                         sampler_loader=sampler_loader,
                         label_sampler_loader=label_sampler_loader,
                         point_sampler_loader=point_sampler_loader,
                         seed=seed,
                         cpus=cpus,
                         normalizer=normalizer,
                         batch_callbacks=batch_callbacks,
                         sample_callbacks=sample_callbacks,
                         log_path=log_path)



class WSIPatchGenerator(BatchGenerator):
    """
    This class generates patches in ordered fashion from a WSI.
    The annotation paths in the data_sourses should be replicates of the image_path.
    The WSIAnnotationParser will make sure that every sampling postion is converted to an annotation.
    """

    @staticmethod
    def block_shaped(arr, nrows, ncols):
        height, _, channels = arr.shape
        return (arr.reshape(height//nrows, nrows, -1, ncols, channels).swapaxes(1, 2).reshape(-1, nrows, ncols, channels))

    @staticmethod
    def unblockshaped(arr, h, w):
        _, nrows, ncols, channels = arr.shape
        return arr.reshape(h//nrows, -1, nrows, ncols, channels).swapaxes(1, 2).reshape(h, w, channels)

    def __init__(self,
                 data_sources,
                 batch_size=1,
                 tile_shape=(1024, 1024, 3),
                 patch_shape=(64, 64, 3),
                 spacing=0.5,
                 shift=(1024, 1024),
                 cpus=1,
                 log_path=None):

        self._patch_shape = patch_shape

        label_map = {'polygon': 1}

        data_set_loader = dataset.DataSetLoader(dataset.DataSet, annotation_types=['polygon'])
        annotation_parser_loader = AnnotationParserLoader(WSIAnnotationParser, tile_shape=tile_shape, shift=shift[0], spacing=spacing)
        index_controller_loader = IndexControllerLoader(controllers.OrderedIndexController)
        point_sampler_loader = PointSamplerLoader(samplers.CenterPointSampler)
        sampler_loader = SamplerLoader(samplers.WSISampler, input_shapes=[tile_shape], spacings=[spacing])

        self._batchgen = super().__init__(data_sources=data_sources,
                                          label_map=label_map,
                                          data_set_loader=data_set_loader,
                                          batch_size=batch_size,
                                          annotation_parser_loader=annotation_parser_loader,
                                          index_controller_loader=index_controller_loader,
                                          point_sampler_loader=point_sampler_loader,
                                          sampler_loader=sampler_loader,
                                          cpus=cpus,
                                          log_path=log_path)

    def batch(self, mode):
        batch = super().batch(mode)
        return np.array(WSIPatchGenerator.block_shaped(batch['x_batch'][0], self._patch_shape[0], self._patch_shape[1]))
    
    
    
    
    
    
    
    
    
