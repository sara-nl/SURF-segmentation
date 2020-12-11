"""
This file defines the BatchMangers and a function to get batch manager.
All definitions in this file should not be used by end users and are only used in the XmlBatchGenerator class

"""


import time
from multiprocessing import Queue
from queue import Full
from threading import Thread
import functools

from .processors import BatchProcess, ControllerProcess
from ..utils import log


def _get_batch_manager(**kwargs):
    """ function that instansiates a batch mananager,
    Returns:  Mulitporcessed manager if cpus  > 4 else return Single processed manager
    """

    if kwargs['cpus'] < 4:
        return _SingleCoreBatchManager(**kwargs)
    return _MultiCoreBatchManager(**kwargs)


def _batch_timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def batch_timer(self, *args, **kwargs):
        start_time = time.perf_counter()
        value = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        self._batch_times_logger.info(f"Finished {func.__name__} {self._batch_index-1} in: {run_time:.4f} secs")
        return value
    return batch_timer


class _SingleCoreBatchManager:
    def __init__(self, **kwargs):
        self._batch_size = kwargs['batch_size']

        self._normalizer = kwargs['normalizer']

        # controllers
        self._sample_controllers = self._init_sample_controllers(**kwargs)

        # batch samplers
        self._batch_sampler = self._init_batch_samplers(**kwargs)

        self._batch_index = 0
        self._logger = kwargs['logging'].get_logger('generator')
        self._batch_times_logger = kwargs['logging'].get_logger('batchtimes')
        self._logger.info(log.system_status())

    @_batch_timer
    def batch(self, mode):
        """ creates a batch given a mode (e.g. training/validation) by computing controller data/sample data.
            A sampler will create the actually samples with the controller data
        """
        controller_batch_data = self._sample_controllers[mode].sample(self._batch_size)
        batch = self._batch_sampler[mode].batch(controller_batch_data, i=self._batch_index)
        self._sample_controllers[mode].update(batch['auxiliaries'])
        self._batch_index += 1
        batch['x_batch'] = self._normalizer(batch['x_batch'])
        return batch

    def reset(self, mode):
        """ Reset funcitonality, (c)urrently not working) """
        pass

    def start(self):
        self._logger.info('batch generator started (single-core)')

    def stop(self):
        for h in self._batch_times_logger.handlers:
            h.close()
        self._batch_index = 0

    def _init_sample_controllers(self, **kwargs):
        # controllers
        return {dataset.mode: kwargs['sampler_controller_loader'](dataset=dataset,
                                                                  label_controller_loader=kwargs['label_controller_loader'],
                                                                  index_controller_loader=kwargs['index_controller_loader'],
                                                                  seed=kwargs['seed'])
                for dataset in kwargs['datasets'].values()}

    def _init_batch_samplers(self, **kwargs):
        # controllers
        return {dataset.mode: kwargs['batch_sampler_loader'](dataset=dataset,
                                                             sampler_loader=kwargs['sampler_loader'],
                                                             patch_sampler_loader=kwargs['patch_sampler_loader'],
                                                             label_sampler_loader=kwargs['label_sampler_loader'],
                                                             point_sampler_loader=kwargs['point_sampler_loader'],
                                                             batch_callbacks=kwargs['batch_callbacks'],
                                                             sample_callbacks=kwargs['sample_callbacks'],
                                                             seed=kwargs['seed'],
                                                             logging=kwargs['logging']) for dataset in kwargs['datasets'].values()}


class _MultiCoreBatchManager:
    def __init__(self, **kwargs):

        self._modes = [dataset.mode for dataset in kwargs['datasets'].values()]

        # batch sampler fields
        self._batch_size = kwargs['batch_size']

        self._normalizer = kwargs['normalizer']

        # system fields
        self._queue_size = kwargs['queue_size']

        self._cpus = kwargs['cpus']

        self._seed = kwargs['seed']

        # set queues
        self._cpu_queues = {mode: Queue(maxsize=self._queue_size) for mode in self._modes}
        self._command_queue = Queue(maxsize=self._queue_size)
        self._controller_queues = [Queue(maxsize=self._queue_size) for cpu_index in range(self._cpus-2)]

        self._batches = {mode: {} for mode in self._modes}
        self._get_index = {mode: 0 for mode in self._modes}
        self._put_index = {mode: 0 for mode in self._modes}

        # set controller process
        self._controller_process = self._init_controller_process(**kwargs)

        # set batch processors
        self._batch_processors = self._init_batch_processors(**kwargs)

        # set manager thread
        self._managers = {mode: Thread(target=self._transfer, daemon=True, args=(mode, )) for mode in self._modes}

        self._batch_index = 0
        self._running = False
        self._logger = kwargs['logging'].get_logger('generator')
        self._batch_times_logger = kwargs['logging'].get_logger('batchtimes')

    def start(self):
        self._running = True

        self._start_processors()

        self._initial_queue_feed()

        self._start_threads()

        self._logger.info('batch generator started (multi-core)')

    @_batch_timer
    def batch(self, mode):
        while self._running:
            index = self._get_index[mode]
            batch = self._batches[mode].get(index)
            if batch:
                self._batch_index = index+1
                self._get_index[mode] += 1
                del self._batches[mode][index]
                self._command_queue.put((self._put_index[mode], mode, batch['auxiliaries']))
                self._put_index[mode] += 1
                return batch

    def stop(self):
        # Stopping
        time.sleep(2)
        self._running = False
        self._send_stop_signals()
        self._flush()
        self._terminate_and_join_processors()
        self._join_manager()
        self._logger.info('batch generator stopped (multi-core)')

    def _init_controller_process(self, **kwargs):
        return ControllerProcess(datasets=kwargs['datasets'],
                                 command_queue=self._command_queue,
                                 controller_queues=self._controller_queues,
                                 batch_size=self._batch_size,
                                 label_controller_loader=kwargs['label_controller_loader'],
                                 index_controller_loader=kwargs['index_controller_loader'],
                                 seed=self._seed)

    def _init_batch_processors(self, **kwargs):
        return [BatchProcess(datasets=kwargs['datasets'],
                             controller_queue=self._controller_queues[cpu_index],
                             cpu_queues=self._cpu_queues,
                             batch_size=self._batch_size,
                             sampler_loader=kwargs['sampler_loader'],
                             batch_sampler_loader=kwargs['batch_sampler_loader'],
                             patch_sampler_loader=kwargs['patch_sampler_loader'],
                             label_sampler_loader=kwargs['label_sampler_loader'],
                             point_sampler_loader=kwargs['point_sampler_loader'],
                             batch_callbacks=kwargs['batch_callbacks'],
                             sample_callbacks=kwargs['sample_callbacks'],
                             seed=self._seed+cpu_index,
                             logging=kwargs['logging']) for cpu_index in range(self._cpus-2)]


    def reset(self, mode):
        """ Reset funcitonality, (c)urrently not working) """
        pass

    def _start_processors(self):
        # start controller
        self._controller_process.start()

        # start up batch processors
        for batch_process in self._batch_processors:
            batch_process.start()

        # verify processors
        for batch_process in self._batch_processors:
            for mode in self._modes:
                start = self._cpu_queues[mode].get()
                self._logger.info(f'Processor: {start}')

    def _initial_queue_feed(self):
       # initial commands
        for mode in self._modes:
            for _ in range(self._queue_size):
                self._command_queue.put((self._put_index[mode], mode, {}))
                self._put_index[mode] += 1

    def _start_threads(self):
        # start manager
        for mode in self._modes:
            self._managers[mode].start()

    def _transfer(self, mode):
        while self._running:
            i, batch = self._cpu_queues[mode].get()
            batch['x_batch'] = self._normalizer(batch['x_batch'])
            self._batches[mode][i] = batch

    def _send_stop_signals(self):
        self._command_queue.put('STOP')
        for i in range(len(self._batch_processors)):
            self._controller_queues[i].put('STOP')

    def _flush(self):
        """ Flushing (should only be called after _send_stop_signals) """
        for _ in iter(self._command_queue.get, 'STOP'):
            continue
        for i in range(len(self._batch_processors)):
            for _ in iter(self._controller_queues[i].get, 'STOP'):
                continue

    def _terminate_and_join_processors(self):
        # joining controller process
        self._controller_process.terminate()
        self._controller_process.join()

        # Joining batch proccesors
        for batch_process in self._batch_processors:
            self._logger.info('stopping batch_process')
            batch_process.terminate()
            batch_process.join()

    def _join_manager(self):
        # make sure manager can join
        for mode in self._modes:
            try:
                self._cpu_queues[mode].put((-1, {'x_batch': [], 'y_batch': [], 'auxiliaries': {}}), False)
            except Full:
                pass
            self._batches[mode] = {}
        for mode in self._modes:
            self._managers[mode].join()
