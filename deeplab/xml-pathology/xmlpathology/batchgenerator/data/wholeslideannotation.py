import numpy as np
import os
from typing import List, Iterator

from shapely import strtree, geometry
from shapely.strtree import STRtree
# from ..data.wholeslideimage import WholeSlideImageASAP
from ..data.wholeslideimage import WholeSlideImageOpenSlide

import abc
import math
import xml.etree.ElementTree as ET
import json
import pdb
import numpy as np
import scipy.ndimage as ndi
from shapely import geometry
from ..utils import shift_coordinates


class Within(object):
    """

    """

    def __init__(self, o):
        self.o = o

    def __lt__(self, other):
        return self.o.buffer(0).within(other.o.buffer(0))


class Polygon(geometry.Polygon):
    def __init__(self,  file_path, index, label_name, label_value, coordinates, holes=[]):
        super().__init__(coordinates, holes)
        self._file_path = file_path
        self._coordinates = coordinates
        self._holes = holes
        self._type = 'polygon'
        self._index = index
        self._label_name = label_name
        self._label_value = label_value
        self._overlapping_annotations = []
        self.density_map = None
        self.density_downsampling_ratio = 0

    def __reduce__(self):
        return (self.__class__, (self._file_path,
                                 self._index,
                                 self._label_name,
                                 self._label_value,
                                 self._coordinates,
                                 self._holes, ))

    @property
    def type(self):
        return self._type

    @property
    def index(self):
        return self._index

    @property
    def label_name(self):
        return self._label_name

    @property
    def label_value(self):
        return self._label_value

    @property
    def bounds(self):
        x1, y1, x2, y2 = super().bounds
        # return [int(x1), int(y1), int(x2), int(y2)]
        return [x1, y1, x2, y2]

    @property
    def size(self):
        xmin, ymin, xmax, ymax = super().bounds
        return math.ceil(xmax-xmin), math.ceil(ymax-ymin)

    @property
    def centroid(self):
        c = super().centroid
        x, y = c.xy
        # return round(x[0]), round(y[0])
        return x[0], y[0]

    @property
    def center(self):
        xmin, ymin, xmax, ymax = super().bounds
        # return round(xmin + (xmax-xmin)/2), round(ymin + (ymax-ymin)/2)
        return xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2

    @property
    def overlapping_annotations(self):
        return self._overlapping_annotations

    @property
    def xy(self):
        return super().xy

    def contains(self, _annotation):
        buffered_annotation = _annotation.buffer(0)
        # check overlapping annotations because if within overlapping than it contains not
        return super().buffer(0).contains(buffered_annotation) and np.all([not annotation.buffer(0).contains(buffered_annotation)
                                                                           for annotation in self._overlapping_annotations if annotation.type != 'dot'])

    def add_overlapping_annotations(self, overlap_annotations):
        self._overlapping_annotations.extend(overlap_annotations)

    def coordinates(self):
        return np.array(self.exterior.xy).T

    def holes(self):
        return [np.array(interior.xy).T for interior in self.interiors]


class Point(geometry.Point):
    def __init__(self, file_path, index, label_name, label_value, coordinates):
        self._file_path = file_path
        self._type = 'dot'
        self._index = index
        self._label_name = label_name
        self._label_value = label_value
        self._coordinates = coordinates
        super().__init__(coordinates)

    def __reduce__(self):
        return (self.__class__, (self._file_path, self._index, self._label_name, self._label_value, self._coordinates, ))

    @property
    def type(self):
        return self._type

    @property
    def index(self):
        return self._index

    @property
    def center(self):
        return int(self.x), int(self.y)

    @property
    def label_name(self):
        return self._label_name

    @property
    def label_value(self):
        return self._label_value

    def coordinates(self):
        return np.array([self.x, self.y])


def create_annotation(file_path, annotation_type, index, label_name, label_value, coordinates, holes=[], **kwargs):
    if annotation_type == 'polygon':
        return Polygon(file_path, index, label_name, label_value, coordinates, holes, **kwargs)

    if annotation_type == 'dot':
        return Point(file_path, index, label_name, label_value, coordinates, **kwargs)


''' Parsers '''


class AnnotationParserLoader:
    def __init__(self, class_, **kwargs):
        self._class = class_
        self._kwargs = kwargs

    def __call__(self, label_map):
        return self._class(label_map, **self._kwargs)


class AnnotationParser:
    def __init__(self, label_map, sort_by='label_map'):
        self._label_map = label_map
        self._sort_by = sort_by

    @property
    def sort_by(self):
        return self._sort_by

    def parse(self, path):
        annotations = []
        opened_annotation = self._open_annotation_file(path)
        for index, annotation_structure in enumerate(self._get_annotation_structures(opened_annotation)):

            label = self._get_label(annotation_structure).strip()
            if label not in self._label_map:
                continue

            annotation_type = self._get_annotation_type(annotation_structure)
            coordinates = self._get_coordinates(annotation_structure)
            holes = self._get_holes(annotation_structure)
            annotations.append(create_annotation(file_path=path,
                                                 annotation_type=annotation_type,
                                                 index=index,
                                                 label_name=label,
                                                 label_value=self._label_map[label],
                                                 coordinates=coordinates,
                                                 holes=holes))
        return annotations


    @abc.abstractmethod
    def _open_annotation_file(self, path):
        pass

    @abc.abstractmethod
    def _get_annotation_structures(self, opened_annotation) -> Iterator:
        pass

    @abc.abstractmethod
    def _get_annotation_type(self, annotation_structure) -> str:
        pass

    @abc.abstractmethod
    def _get_coordinates(self, annotation_structure) -> List:
        pass

    @abc.abstractmethod
    def _get_holes(self, annotation_structure):
        pass

    @abc.abstractmethod
    def _get_label(self, annotation_structure) -> str:
        pass


class WSIAnnotationParser(AnnotationParser):
    def __init__(self, label_map, sort_by='label_map', tile_shape=(256, 256), shift=256, spacing=0.5):
        super().__init__(label_map=label_map, sort_by=sort_by)
        self._tile_shape = tile_shape
        self._shift = shift
        self._pixel_spacing = spacing
        self._get_label = self._get_annotation_type

    def _open_annotation_file(self, path):
        image = WholeSlideImageASAP(image_path=path)
        downsampling = image.get_downsampling_from_spacing(self._pixel_spacing)
        x_dims, y_dims = image.shapes[image.get_level_from_spacing(self._pixel_spacing)]
        image.close()
        del image
        return x_dims, y_dims, downsampling

    def _get_annotation_structures(self, opened_annotation):
        x_dims, y_dims, downsampling = opened_annotation
        for y_pos in range(0, y_dims, self._tile_shape[0]):
            for x_pos in range(0, x_dims, self._tile_shape[1]):
                annotation_structure = {'x_pos': int(x_pos*downsampling),
                                        'y_pos': int(y_pos*downsampling),
                                        'downsampling': downsampling,
                                        'annotation_type': 'polygon'}

                yield annotation_structure

    def _get_annotation_type(self, annotation_structure) -> str:
        return annotation_structure['annotation_type']

    def _get_coordinates(self, annotation_structure) -> List:
        x_pos = annotation_structure['x_pos']
        y_pos = annotation_structure['y_pos']
        downsampling = annotation_structure['downsampling']

        return [(x_pos, y_pos),
                (x_pos, y_pos+self._tile_shape[1]*downsampling),
                (x_pos+self._tile_shape[0]*downsampling, y_pos+self._tile_shape[1]*downsampling),
                (x_pos+self._tile_shape[0]*downsampling, y_pos)]

    def _get_holes(self, annotation_structure):
        return []

class AsapAnnotationParser(AnnotationParser):
    def __init__(self, label_map, sort_by='label_map'):
        super().__init__(label_map=label_map, sort_by=sort_by)

    def _open_annotation_file(self, path):
        tree = ET.parse(path)
        return tree.getroot()

    def _get_annotation_structures(self, opened_annotation):
        for parent in opened_annotation:
            for child in parent:
                if child.tag == 'Annotation':
                    yield child

    def _get_annotation_type(self, annotation_structure) -> str:
        annotation_type = annotation_structure.attrib.get('Type').lower()
        if annotation_type.lower() in ['polygon', 'rectangle', 'dot', 'spline']:
            annotation_type = annotation_type.lower()
            if annotation_type != 'dot':
                annotation_type = 'polygon'
            return annotation_type
        raise ValueError(f'unsupported annotation type in {annotation_structure}')

    def _get_label(self, annotation_structure) -> str:
        if self._get_annotation_type(annotation_structure) in self._label_map:
            return self._get_annotation_type(annotation_structure)
        return annotation_structure.attrib.get('PartOfGroup').lower()

    def _get_coordinates(self, annotation_structure) -> List:
        coordinate_structure = annotation_structure[0]
        return [(float(coord.get('X').replace(',', '.')), float(coord.get('Y').replace(',', '.'))) for coord in coordinate_structure]

    def _get_holes(self, annotation_structure):
        return []


class VirtumAnnotationParser(AnnotationParser):
    def __init__(self, label_map, sort_by='label_map'):
        super().__init__(label_map=label_map, sort_by=sort_by)

    def _open_annotation_file(self, path):
        tree = ET.parse(path)
        vsannotations = []
        name_to_group = {}
        for parent in tree.getroot():
            for child in parent:
                if child.tag == 'Annotation':
                    vsannotations.append(child)
                if child.tag == 'Group':
                    group = child
                    name = group.attrib.get('Name')
                    if 'tissue' not in name and 'holes' not in name:
                        partofgroup = group.attrib.get('PartOfGroup')
                        if partofgroup != 'None':
                            name_to_group[group.attrib.get('Name')] = group.attrib.get('PartOfGroup')

        opened_annotation = []
        for annotation in vsannotations:
            label_reference = annotation.attrib.get('PartOfGroup').split('_')[0]
            tissue = annotation.attrib.get('PartOfGroup').split('_')[1] == 'tissue'
            label = name_to_group[label_reference] if tissue else 'hole'
            coordinates = []
            for coords in annotation:
                coordinates = [(float(coord.get('X').replace(',', '.')), float(coord.get('Y').replace(',', '.'))) for coord in coords]
            if label == 'hole':
                opened_annotation[-1]['holes'].append(coordinates)
            else:
                opened_annotation.append({'type': 'polygon', 'label': label, 'coordinates': coordinates, 'holes': []})

        return opened_annotation

    def _get_annotation_structures(self, opened_annotation):
        for annotation in opened_annotation:
            yield annotation

    def _get_annotation_type(self, annotation_structure) -> str:
        return annotation_structure['type']

    def _get_label(self, annotation_structure) -> str:
        return annotation_structure['label']

    def _get_coordinates(self, annotation_structure) -> List:
        return annotation_structure['coordinates']

    def _get_holes(self, annotation_structure):
        return annotation_structure['holes']


# class VirtumAnnotationParser(AnnotationParser):
#     def __init__(self, label_map, sort_by):
#         super().__init__(label_map=label_map, sort_by=sort_by)

#     def _get_annotation_structures(self, opened_annotation):

#     def parse(self, path):
#         annotations = []
#         vsannotations = []
#         name_to_group = {}
#         tree = ET.parse(path)
#         root = tree.getroot()
#         for parent in root:
#             for child in parent:
#                 if child.tag == 'Annotation':
#                     vsannotations.append(child)
#                 if child.tag == 'Group':
#                     group = child
#                     name = group.attrib.get('Name')
#                     if 'tissue' not in name and 'holes' not in name:
#                         partofgroup = group.attrib.get('PartOfGroup')
#                         if partofgroup != 'None':
#                             name_to_group[group.attrib.get('Name')] = group.attrib.get('PartOfGroup')

#         temp_annotations = []
#         idx = 0
#         for annotation in vsannotations:
#             label_reference = annotation.attrib.get('PartOfGroup').split('_')[0]
#             tissue = annotation.attrib.get('PartOfGroup').split('_')[1] == 'tissue'
#             label = name_to_group[label_reference] if tissue else 'hole'
#             coordinates = []
#             for coords in annotation:
#                 coordinates = [(float(coord.get('X').replace(',', '.')), float(coord.get('Y').replace(',', '.'))) for coord in coords]
#             if label == 'hole':
#                 temp_annotations[-1]['holes'].append(coordinates)
#             else:
#                 temp_annotations.append({'type': 'polygon', 'index': idx, 'label': label, 'coordinates': coordinates, 'holes': []})
#                 idx += 1

#         for annotation in temp_annotations:
#             annotations.append(create_annotation(file_path=path,
#                                                  annotation_type=annotation['type'],
#                                                  index=annotation['index'],
#                                                  label_name=annotation['label'],
#                                                  label_value=self._label_map[annotation['label']],
#                                                  coordinates=annotation['coordinates'],
#                                                  holes=annotation['holes']))
#         return annotations




# class JsonAnnotationParser(AnnotationParser):
#     def __init__(self, label_map, sort_by, rect_margin=0):
#         super().__init__(label_map=label_map, sort_by='within')
#         self.rect_margin = rect_margin

#     def parse(self, path):
#         index = 0
#         annotations = []
#         with open(path) as json_file:
#             print('json file:', path)
#             data = json.load(json_file)
#             for d in data:
#                 elements = d['annotation']['elements']
#                 for element in elements:
#                     if 'label' not in element:
#                         continue
#                     label_name = element['label']['value']
#                     # parse roi
#                     if element['type'] == 'rectangle':
#                         center = element['center']
#                         width = element['width']
#                         height = element['height']

#                         x1, y1 = center[0] - width//2 + self.rect_margin, center[1] - height//2 + self.rect_margin
#                         x2, y2 = x1 + element['width'] - self.rect_margin*2, y1 + element['height'] - self.rect_margin*2
#                         coordinates = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
#                         if 'rotation' in element and element['rotation'] != 0:
#                             rotation = element['rotation']
#                             p = geometry.Polygon(coordinates)
#                             from shapely import affinity
#                             rotated = affinity.rotate(p, rotation, use_radians=True)
#                             coordinates = list(rotated.exterior.coords)[:-1]  # first=last point

#                     # parse polygon
#                     else:
#                         points = element['points']
#                         coordinates = [point[:2] for point in points]

#                     fill_color = element.get('fillColor', None)
#                     line_color = element.get('lineColor', None)  # todo convert 'rgba(0, 0, 0, 0)' to hex
#                     annotations.append(create_annotation(xml_path=path,
#                                                          annotation_type='polygon',
#                                                          index=index,
#                                                          label_name=label_name,
#                                                          label_value=self._label_map[label_name],
#                                                          coordinates=coordinates,
#                                                          fill_color=fill_color,
#                                                          line_color=line_color))
#                     index += 1
#         return annotations



class QuPathParser(AnnotationParser):
    def __init__(self, label_map, sort_by='label_map'):
        super().__init__(label_map=label_map, sort_by=sort_by)

    def _open_annotation_file(self, path):
        with open(path) as json_file:
            json_annotations = json.load(json_file)
        return json_annotations

    def _get_annotation_structures(self, opened_annotation) -> Iterator:
        for json_annotation in opened_annotation:
            if json_annotation['geometry']['type'].lower() == 'polygon':
                yield json_annotation

    def _get_annotation_type(self, annotation_structure) -> str:
        return 'polygon'

    def _get_coordinates(self, annotation_structure) -> List:
        return annotation_structure['geometry']['coordinates'][0]

    def _get_label(self, annotation_structure) -> str:
        return annotation_structure['properties']['classification']['name'].lower()

    def _get_holes(self, annotation_structure):
        return []

class WholeSlideAnnotation:
    """
    This class contains all annotations of an image.

    """

    # fix for strtree segmentation fault bug
    STREE = {}

    def __init__(self,
                 data_source_id,
                 image_path,
                 annotation_path,
                 label_map,
                 annotation_types=('polygon', ),
                 annotation_parser_loader=AnnotationParserLoader(class_=AsapAnnotationParser, sort_by='label_map')):

        self._data_source_id = data_source_id

        self._annotation_types = annotation_types

        self._annotation_path = annotation_path

        self._image_path = image_path

        self._label_map = label_map

        self._annotation_parser = annotation_parser_loader(self._label_map)

        self._annotations = self._annotation_parser.parse(annotation_path)

        self.annotations = [annotation for annotation_type in self._annotation_types
                            for annotation in self._annotations
                            if annotation.type == annotation_type]

        if self._annotation_parser.sort_by == 'label_map':
            self._annotations = sorted(self._annotations, key=lambda item: self._label_map[item.label_name])
        if self._annotation_parser.sort_by == 'within':
            self._annotations = sorted(self._annotations, key=Within, reverse=True)

        # self._set_overlapping_annotations()

        WholeSlideAnnotation.STREE[data_source_id] = STRtree(self._annotations)

        if os.path.exists(self._image_path):
            self._spacings, self._downsamplings = self._get_image_info()
        else:
            raise ValueError(f'image: {self._image_path} does not exists')

    @property
    def data_source_id(self):
        return self._data_source_id

    @property
    def annotation_path(self):
        return self._annotation_path

    @property
    def image_path(self):
        return self._image_path

    @property
    def label_map(self):
        return self._label_map

    @property
    def counts(self):
        return len(self.annotations)

    @property
    def count_per_class(self):
        cpc = {label_name: 0 for label_name in self._label_map}
        for annotation in self.annotations:
            cpc[annotation.label_name] += 1
        return cpc

    @property
    def pixels(self):
        return int(sum([annotation.area for annotation in self.annotations]))

    @property
    def pixels_per_class(self):
        ppc = {label_name: 0 for label_name in self._label_map}
        for annotation in self.annotations:
            ppc[annotation.label_name] += int(annotation.area)
        return ppc

    @property
    def spacings(self):
        return self._spacings

    @property
    def downsamplings(self):
        return self._downsamplings

    def _set_overlapping_annotations(self):
        for annotation_index, annotation in enumerate(self._annotations[:-1]):
            if annotation.type == 'polygon':
                overlap_tree = strtree.STRtree(annotation for annotation in self._annotations[annotation_index+1:])
                annotation.add_overlapping_annotations(overlap_tree.query(annotation))


    def _set_densities(self, density_values):
        for annotation in self.annotations:
            annotation.load_density(density_values[0], density_values[1])

    def _get_image_info(self):
        # image = WholeSlideImageASAP(self._image_path)
        image = WholeSlideImageOpenSlide(self._image_path)
        spacings, downsamplings = image.spacings, image.downsamplings
        image.close()
        image = None
        return spacings, downsamplings

    def get_downsampling_from_level(self, level: int) -> float:
        return self.downsamplings[level]

    def get_level_from_spacing(self, spacing: float) -> int:
        spacing_margin = spacing*(1/10)
        for level, s in enumerate(self.spacings):
            if abs(s - spacing) <= spacing_margin:
                return level
        raise ValueError('imageannotaiton spacing error')

    def get_ratio(self, spacing: float) -> float:
        return self.get_downsampling_from_level(self.get_level_from_spacing(spacing))

    def select_annotations(self, center_x, center_y, width, height):
        box = geometry.box(center_x - width//2, center_y - height//2, center_x + width//2, center_y + height//2)
        boxint = geometry.box(int(center_x - width//2), int(center_y - height//2), int(center_x + width//2), int(center_y + height//2))
        
        result = WholeSlideAnnotation.STREE[self._data_source_id].query(box)
        resultint = WholeSlideAnnotation.STREE[self._data_source_id].query(boxint)
        if not (len(result) or len(resultint)):
            print("Empty")
        pdb.set_trace()
        if self._annotation_parser.sort_by == 'label_map':
            return sorted(result, key=lambda item: self._label_map[item.label_name])
        if self._annotation_parser.sort_by == 'within':
            return sorted(result, key=Within, reverse=True)
        
        return result
    
    
    