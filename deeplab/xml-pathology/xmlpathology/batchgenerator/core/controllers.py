"""
Module docstring

"""

import abc
import itertools
from typing import List, Type, Callable, Union, Dict
from collections.abc import Iterator
import pdb
import numpy as np


# Abstract Controllers #


class Controller(Iterator):
    def __init__(self, seed: int):
        self._seed = seed
        np.random.seed(seed)

    def set_seed(self):
        np.random.seed(self._seed)

    @abc.abstractmethod
    def __next__(self) -> Union[str, Callable]:
        pass

    @abc.abstractmethod
    def update(self, auxiliaries: Dict[str, Dict]):
        pass


class LabelController(Controller):
    def __init__(self, labels: List[str], seed: int):
        super().__init__(seed=seed)
        self._labels = labels
        self._size = len(labels)

    @abc.abstractmethod
    def __next__(self) -> str:
        pass

    @abc.abstractmethod
    def update(self, auxiliaries: Dict[str, Dict]):
        pass

class IndexController(Controller):
    def __init__(self, counts_per_label: Dict[str, int], seed: int):
        super().__init__(seed)
        self._counts_per_label = counts_per_label

    def __next__(self) -> Callable:
        return self._next

    @abc.abstractmethod
    def _next(self, label: str) -> int:
        pass

    @abc.abstractmethod
    def update(self, auxiliaries: Dict[str, Dict]):
        pass


# Label Controllers #


class RandomLabelController(LabelController):

    def __init__(self, labels: List[str], seed: int = 123):
        super().__init__(labels=labels, seed=seed)

    def __next__(self) -> str:
        return self._labels[np.random.randint(self._size)]

    def update(self, auxiliaries: Dict[str, Dict]):
        pass


class StrictLabelController(LabelController):
    def __init__(self, labels, seed=123):
        super().__init__(labels, seed)
        self._labels_cycle = itertools.cycle(self._labels)
        self.reset()

    def __next__(self):
        return next(self._labels_cycle)

    def reset(self):
        self._labels_cycle = itertools.cycle(self._labels)

    def update(self, auxiliaries: Dict[str, Dict]):
        pass

class BalancedLabelController(LabelController):

    def __init__(self, labels, seed=123):
        super().__init__(labels, seed)
        np.random.shuffle(self._labels)
        self._labels_cycle = iter(self._labels)
        self.reset()

    def __next__(self):
        try:
            return next(self._labels_cycle)
        
        except StopIteration:
            self.reset()
            return next(self._labels_cycle)

    def reset(self):
        np.random.shuffle(self._labels)
        self._labels_cycle = iter(self._labels)

    def update(self, auxiliaries: Dict[str, Dict]):
        pass


class PixelLabelController(LabelController):

    def __init__(self, labels: List[str], seed: int = 123):
        super().__init__(labels, seed)
        self._pixel_count_per_label = {label: 1 for label in self._labels}

    def __next__(self):
        total = sum(self._pixel_count_per_label.values())
        inverse_ratios = {label: 1 / (value / total) for label, value in self._pixel_count_per_label.items()}
        inverse_total = sum(inverse_ratios.values())
        ratios = {label: value / inverse_total for label, value in inverse_ratios.items()}
        return np.random.choice(list(ratios.keys()), p=list(ratios.values()))

    def update(self, auxiliaries: Dict[str, Dict]):
        try:
            for label, counts in auxiliaries['pixel_count'].items():
                self._pixel_count_per_label[label] += counts
        except KeyError:
            raise ValueError('no \'pixel_count\' key found in auxiliaries, consider using the PixelCounter callback when using the PixelLabelController')


class FslRandomLabelController(LabelController):

    def __init__(self, labels, seed=123):
        super().__init__(labels=labels, seed=seed)

    def __next__(self):
        return self._next

    def _next(self, n_way):
        np.random.shuffle(self._labels)
        return self._labels[:n_way]

    def update(self, auxiliaries: Dict[str, Dict]):
        pass

class ProbabilityLabelController:
    pass


class OrderedIndexController(IndexController):
    def __init__(self, counts_per_label, seed=123):
        super().__init__(counts_per_label, seed)
        self._counters={label: 0 for label in self._counts_per_label.keys()}
        self._indexes={label: list(range(counts)) for label, counts in self._counts_per_label.items()}
        self.reset()

    def __next__(self):
        return self._next

    def _next(self, label):
        index=self._indexes[label][self._counters[label]]
        self._counters[label] += 1
        if self._counters[label] == self._counts_per_label[label]:
            self._reset_label(label)
        return index

    def _reset_label(self, label):
        self._counters[label]=0

    def update(self, auxiliaries: Dict[str, Dict]):
        pass

    def reset(self):
        self.set_seed()
        for label in self._counts_per_label.keys():
            self._reset_label(label)


class BalancedIndexController(IndexController):

    def __init__(self, counts_per_label, seed=123):
        super().__init__(counts_per_label, seed)
        self._counters={label: 0 for label in self._counts_per_label.keys()}
        self._indexes={label: list(range(counts)) for label, counts in self._counts_per_label.items()}
        self.reset()

    def __next__(self):
        return self._next

    def _next(self, label):
        index=self._indexes[label][self._counters[label]]
        self._counters[label] += 1
        if self._counters[label] == self._counts_per_label[label]:
            self._reset_label(label)
        return index

    def update(self, auxiliaries: Dict[str, Dict]):
        pass

    def _reset_label(self, label):
        self._counters[label]=0
        np.random.shuffle(self._indexes[label])

    def reset(self):
        self.set_seed()
        for label in self._counts_per_label.keys():
            self._reset_label(label)


class FslRandomIndexController(IndexController):

    def __init__(self, counts_per_label, seed=123):
        super().__init__(counts_per_label, seed)
        self._indexes={label: list(range(counts)) for label, counts in self._counts_per_label.items()}
        self.reset()

    def __next__(self):
        return self._next

    def _next(self, label, counts):
        np.random.shuffle(self._indexes[label])
        return self._indexes[label][:counts]

    def update(self, auxiliaries: Dict[str, Dict]):
        pass

    def reset(self):
        pass



# Controller Loaders #


class LabelControllerLoader:
    """
    LabelController loader

    """

    def __init__(self, class_: Type[LabelController], **kwargs) -> None:
        self.class_=class_
        self._kwargs=kwargs

    def __call__(self, labels: List[str], seed: int) -> LabelController:
        return self.class_(labels=labels, seed=seed, **self._kwargs)


class IndexControllerLoader:
    """
    IndexController loader

    """

    def __init__(self, class_: Type[IndexController], **kwargs) -> None:
        self.class_=class_
        self._kwargs=kwargs

    def __call__(self, counts_per_label, seed):
        return self.class_(counts_per_label=counts_per_label, seed=seed, **self._kwargs)


class SamplerControllerLoader():
    def __init__(self, class_, **kwargs):
        self._class=class_
        self._kwargs=kwargs

    def __call__(self, dataset, label_controller_loader, index_controller_loader, seed):
        return self._class(dataset=dataset,
                           label_controller_loader=label_controller_loader,
                           index_controller_loader=index_controller_loader,
                           seed=seed, **self._kwargs)



# Sample Conrollers #


class SamplerController:
    def __init__(self,
                 dataset,
                 label_controller_loader=LabelControllerLoader(BalancedLabelController),
                 index_controller_loader=IndexControllerLoader(BalancedIndexController),
                 seed=123):

        # set dataset
        self._dataset = dataset
        np.random.seed(seed)

        # set controllers
        self._label_controller = label_controller_loader(labels=self._dataset.labels, seed=seed)
        self._index_controller = index_controller_loader(counts_per_label=self._dataset.counts_per_label, seed=seed)

    @property
    def mode(self):
        return self._dataset.mode

    def reset(self):
        self._label_controller.reset()
        self._index_controller.reset()

    def update(self, auxiliaries: Dict[str, Dict]):
        self._label_controller.update(auxiliaries)
        self._index_controller.update(auxiliaries)

    def sample(self, batch_size):
        samples=[]
        for _ in range(batch_size):
            # get next label
            label=next(self._label_controller)

            # get next index of label
            index=next(self._index_controller)(label)

            # get new sample to samples
            sample=self._dataset.samples[label][index]

            # add new sample to samples
            samples.append((sample['image_annotation_index'], sample['annotation_index']))

        return samples


class FslSamplerController(SamplerController):
    def __init__(self,
                 dataset,
                 n_shots=1,
                 n_way=5,
                 n_queries=4,
                 label_controller_loader=LabelControllerLoader(FslRandomLabelController),
                 index_controller_loader=IndexControllerLoader(FslRandomIndexController),
                 seed=123):

        super().__init__(dataset, label_controller_loader, index_controller_loader, seed)
        # check if we can sample by n_way
        if len(self._dataset.labels) < n_way:
            raise ValueError(f'FslSampleCOntroller: number of classes ({len(self._dataset.info.labels)}) < n_way ({n_way})')

        # check if we can sample shots_n_queries examples
        for label, counts in self._dataset.counts_per_label.items():
            if 0 < counts < n_shots+n_queries:
                raise ValueError(f'FslSampleCOntroller: {label} (n={counts})  does not have {n_shots+n_queries} samples')

        self._n_way = n_way
        self._n_shots = n_shots
        self._n_queries = n_queries

    def sample(self, batch_size):
        samples = []
        for _ in range(batch_size):
            # get next label
            labels = self._label_controller.next(self._n_way)

            for label in labels:
                # get next index of label
                indexes = self._index_controller.next(label, self._n_shots+self._n_queries)

                for index in indexes:
                    # get new sample to samples
                    sample = self._dataset.samples[label][index]

                    # add new sample to samples
                    samples.append((sample['image_annotation_index'], sample['annotation_index']))

        return samples
