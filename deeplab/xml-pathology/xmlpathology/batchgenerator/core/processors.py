from multiprocessing import Process
import os

import numpy as np

from ..data import dataset
from .controllers import (BalancedIndexController, BalancedLabelController,
                          IndexControllerLoader, LabelControllerLoader,
                          SamplerController, SamplerControllerLoader)
from .samplers import (BatchSampler, BatchSamplerLoader, LabelSamplerLoader,
                       PatchSampler, PatchSamplerLoader, PointSamplerLoader,
                       RandomPointSampler, Sampler, SamplerLoader,
                       SegmentationLabelSampler)
from ..utils import log

class ControllerProcess(Process):
    def __init__(self,
                 datasets,
                 command_queue,
                 controller_queues,
                 batch_size=1,
                 sampler_controller_loader=SamplerControllerLoader(SamplerController),
                 label_controller_loader=LabelControllerLoader(BalancedLabelController, seed=123),
                 index_controller_loader=IndexControllerLoader(BalancedIndexController, seed=123),
                 seed=123):

        super().__init__()

        self.daemon = True

        self._seed = seed
        np.random.seed(self._seed)

        # construction to make spawn mode work (otherwise sstree will hang)
        self._datasets, self._datasets_ = None, datasets
        # self._datasets= datasets

        self._command_queue = command_queue
        self._controller_queues = controller_queues

        self._batch_size = batch_size

        self._seed = seed

        # controllers
        self._controllers = None

        self._sampler_controller_loader = sampler_controller_loader
        self._label_controller_loader = label_controller_loader
        self._index_controller_loader = index_controller_loader

    def run(self):
        np.random.seed(self._seed)
        # controllers
        self._datasets = dataset.create_new_datasets(self._datasets_)

        self._controllers = {dataset.mode: self._sampler_controller_loader(dataset=dataset,
                                                                            label_controller_loader=self._label_controller_loader,
                                                                            index_controller_loader=self._index_controller_loader,
                                                                            seed=self._seed) for dataset in self._datasets.values()}

        self._run()

    def _run(self):
        for command in iter(self._command_queue.get, 'STOP'):
            i, mode, auxiliaries = command
            if auxiliaries:
                self._controllers[mode].update(auxiliaries)
            controller_batch_data = self._controllers[mode].sample(self._batch_size)
            self._controller_queues[i%len(self._controller_queues)].put((i, mode, controller_batch_data))
        self._command_queue.put('STOP')

    def reset(self, mode):
        self._controllers[mode].reset()


class BatchProcess(Process):
    def __init__(self,
                 datasets,
                 controller_queue,
                 cpu_queues,
                 batch_size=1,
                 sampler_loader=SamplerLoader(class_=Sampler, input_shapes=[[256, 256, 3]], spacings=[0.5]),
                 batch_sampler_loader=BatchSamplerLoader(class_=BatchSampler),
                 patch_sampler_loader=PatchSamplerLoader(class_=PatchSampler),
                 label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                 point_sampler_loader=PointSamplerLoader(class_=RandomPointSampler, strict_point_sampling=False),
                 batch_callbacks=None,
                 sample_callbacks=None,
                 seed=123,
                 logging=log.Logger()):

        super().__init__()
        self.daemon = True

        self._seed = seed
        np.random.seed(self._seed)

        # data fields
        # construction to make spawn mode work (otherwise sstree will hang)
        self._datasets, self._datasets_ = None, datasets
        # self._datasets = datasets

        self._controller_queue = controller_queue
        self._cpu_queues = cpu_queues

        # batch sampler fields
        self._batch_size = batch_size
        self._seed = seed
        self._batch_samplers = None
        self._batch_sampler_loader = batch_sampler_loader
        self._sampler_loader = sampler_loader
        self._patch_sampler_loader = patch_sampler_loader
        self._label_sampler_loader = label_sampler_loader
        self._point_sampler_loader = point_sampler_loader
        self._batch_callbacks = batch_callbacks
        self._sample_callbacks = sample_callbacks
        self._logging = logging

    def run(self):
        np.random.seed(self._seed)

        self._datasets = dataset.create_new_datasets(self._datasets_)

        self._batch_samplers = {dataset.mode: self._batch_sampler_loader(dataset=dataset,
                                                                  sampler_loader=self._sampler_loader,
                                                                  patch_sampler_loader=self._patch_sampler_loader,
                                                                  label_sampler_loader=self._label_sampler_loader,
                                                                  point_sampler_loader=self._point_sampler_loader,
                                                                  batch_callbacks=self._batch_callbacks,
                                                                  sample_callbacks=self._sample_callbacks,
                                                                  seed=self._seed,
                                                                  logging=self._logging) for dataset in self._datasets.values()}

        for mode in self._cpu_queues:
            self._cpu_queues[mode].put((os.getpid(), mode, 'Start'))

        self._run()

    def _run(self):
        for controller_message in iter(self._controller_queue.get, 'STOP'):
            i, mode, controller_data = controller_message
            batch = self._batch_samplers[mode].batch(controller_data, i=i)
            self._cpu_queues[mode].put((i, batch))
        self._controller_queue.put('STOP')

    def reset(self, mode):
        self._batch_samplers[mode].reset()
