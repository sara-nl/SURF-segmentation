from .wholeslideimage import WholeSlideImageOpenSlide
from .wholeslideannotation import WholeSlideAnnotation, AnnotationParserLoader, AsapAnnotationParser
import pdb

def create_new_datasets(datasets):
    return {dataset.mode: DataSet(mode=dataset.mode,
                                  data_source=dataset.data_source,
                                  label_map=dataset.label_map,
                                  annotation_parser_loader=dataset.annotation_parser_loader,
                                  open_images_ahead=dataset.open_images_ahead,
                                  annotation_types=dataset.annotation_types) for dataset in datasets.values()}


class DataSetLoader:

    def __init__(self, class_, **kwargs):
        self._class = class_
        self._kwargs = kwargs

    def __call__(self, mode, data_source, label_map, annotation_parser_loader):
        return self._class(mode=mode,
                           data_source=data_source,
                           label_map=label_map,
                           annotation_parser_loader=annotation_parser_loader,
                           **self._kwargs)


class DataSet():
    def __init__(self,
                 mode,
                 data_source,
                 label_map,
                 annotation_parser_loader=AnnotationParserLoader(class_=AsapAnnotationParser, sort_by='label_map'),
                 annotation_types=('polygon', ),
                 open_images_ahead=False):

        self._mode = mode
        self._data_source = data_source
        self._data_source_map = {data_source_id: source for data_source_id, source in enumerate(self._data_source)}
        self._label_map = {key.lower(): value for key, value in label_map.items()}
        self._annotation_types = annotation_types
        self._annotation_parser_loader = annotation_parser_loader

        self._open_images_ahead = open_images_ahead
        self._images = self._open_images() if self._open_images_ahead else {}
        self._image_annotations = self._open_image_annotations()
        self._samples = self._init_samples()

    @property
    def mode(self):
        return self._mode

    @property
    def annotation_parser_loader(self):
        return self._annotation_parser_loader

    @property
    def data_source(self):
        return self._data_source

    @property
    def data_source_map(self):
        return self._data_source_map

    @property
    def label_map(self):
        return self._label_map

    @property
    def open_images_ahead(self):
        return self._open_images_ahead

    @property
    def images(self):
        return self._images

    @property
    def image_annotations(self):
        return self._image_annotations

    @property
    def annotation_types(self):
        return self._annotation_types

    @property
    def samples(self):
        return self._samples

    def get_image_path(self, data_source_id):
        return self._data_source_map[data_source_id]['image_path']

    def get_mask_path(self, data_source_id):
        mask_path = self._data_source_map[data_source_id].get('mask_path', None)
        if mask_path:
            if mask_path.lower() == 'none' or mask_path == '':
                return None
        return mask_path

    def _open_image_annotations(self):
        image_annotations = []
        for data_source_id, data_source in self._data_source_map.items():
            image_annotations.append(WholeSlideAnnotation(data_source_id=data_source_id,
                                                          annotation_path=data_source['annotation_path'],
                                                          image_path=data_source['image_path'],
                                                          label_map=self._label_map,
                                                          annotation_parser_loader=self._annotation_parser_loader,
                                                          annotation_types=self._annotation_types))
        return image_annotations

    def _init_samples(self):
        samples = {}
        for image_annotation_index, image_annotation in enumerate(self._image_annotations):
            for annotation_index, annotation in enumerate(image_annotation.annotations):
                samples.setdefault(annotation.label_name, []).append({'image_annotation_index': image_annotation_index,
                                                                      'annotation_index': annotation_index})
        return samples

    def _open_images(self):
        images = {}
        for data_source in self._data_source_map.values():
            images[data_source['image_path']] = WholeSlideImageOpenSlide(image_path=data_source['image_path'])
        return images

    def close_images(self):
        for image in self._images.values():
            image.close()
            del image
        self._images = {}

    @property
    def labels(self):
        return list(self.samples.keys())

    @property
    def counts(self):
        return sum([image_annotation.counts for image_annotation in self.image_annotations])

    @property
    def counts_per_label(self):
        counts_per_label_ = {label: 0 for label in self.labels}
        for image_annotation in self.image_annotations:
            for label, count in image_annotation.count_per_class.items():
                if label in counts_per_label_:
                    counts_per_label_[label] += count
        return counts_per_label_

    @property
    def counts_per_image(self):
        return {image_annotation.image_path: image_annotation.counts for image_annotation in self.image_annotations}

    @property
    def counts_per_label_per_image(self):
        counts_per_label_per_image_ = {}
        for image_annotation in self.image_annotations:
            counts_per_label_per_image_[image_annotation.image_path] = image_annotation.count_per_class
        return counts_per_label_per_image_

    @property
    def pixels(self):
        return int(sum([image_annotation.pixels for image_annotation in self.image_annotations]))

    @property
    def pixels_per_label(self):
        pixels_per_label_ = {label: 0 for label in self.labels}
        for image_annotation in self.image_annotations:
            for label, count in image_annotation.pixels_per_class.items():
                pixels_per_label_[label] += count
        return pixels_per_label_

    @property
    def pixels_per_image(self):
        return {image_annotation.image_path: image_annotation.pixels for image_annotation in self.image_annotations}

    @property
    def pixels_per_label_per_image(self):
        pixels_per_label_per_image_ = {}
        for image_annotation in self.image_annotations:
            pixels_per_label_per_image_[image_annotation.image_path] = image_annotation.pixels_per_class
        return pixels_per_label_per_image_
