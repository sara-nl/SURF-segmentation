import glob
import os
import time
from datetime import datetime

from xml.etree import ElementTree as ET
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment, tostring, ElementTree


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import colors
import pdb

# if not 'DISPLAY' in os.environ:
# matplotlib.use('agg')


EXTENSIONS = ('tif', 'tiff', 'mrxs', 'svs')


def get_input_files(data_source):
    if os.path.isdir(data_source):
        pass
        # read directory for files
    elif os.path.isfile(data_source):
        if data_source.lower().endswith(EXTENSIONS):
            pass


def set_output_file(filename, log_path=None, suffix='', output_folder='', overwrite=False):
    if log_path:
        filename = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(log_path, output_folder)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        filename = os.path.join(output_path, filename)

    if suffix != '':
        filename = os.path.splitext(filename)[0] + suffix

    if os.path.exists(filename) and not overwrite:
        print('WARNING: file already exists and overwrite is False')
        filename_parts = os.path.splitext(filename)
        filename = filename_parts[0] + datetime.now().strftime("%d%m%Y%H%M%S") + filename_parts[1]
        print('creating datetime stamped file:', filename)

    return filename

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def one_hot_encoding(mask, label_map):
    """Encodes mask into one hot encoding."""
    ncols = max(label_map.values())+1
    out = np.zeros((mask.size, ncols), dtype=np.uint8)
    out[np.arange(mask.size), mask.ravel()] = 1
    out.shape = mask.shape + (ncols,)
    if 0 in label_map.values():
        return out
    return out[..., 1:]


def one_hot_decoding(mask, base=-1):
    """decode one hot encoding"""
    xs, ys, lbl = np.where(mask)
    new_mask = np.zeros((mask.shape[0], mask.shape[1]))
    new_mask[xs, ys] = lbl.astype(int) + 1
    return new_mask + base


def shift_coordinates(coordinates, center_x, center_y, width, height, ratio):
    coordinates -= np.array([center_x, center_y])
    coordinates /= ratio
    coordinates += np.array([width//2, height//2])
    return coordinates


def normalize(input_):
    _type = type(input_)
    if _type == np.ndarray:
        return input_ / 255.0
    return _type(np.array(input_) / 255.0)


def fit_data(data, output_shape):
    cropx = (data.shape[0] - output_shape[0]) // 2
    cropy = (data.shape[1] - output_shape[1]) // 2

    if len(data.shape) == 2:
        return data[cropx:-cropx, cropy:-cropy]
    if len(data.shape) == 3:
        return data[cropx:-cropx, cropy:-cropy, :]
    if len(data.shape) == 4:
        cropx = (data.shape[1] - output_shape[0]) // 2
        cropy = (data.shape[2] - output_shape[1]) // 2
        return data[:, cropx:-cropx, cropy:-cropy, :]
    if len(data.shape) == 5:
        cropx = (data.shape[2] - output_shape[0]) // 2
        cropy = (data.shape[3] - output_shape[1]) // 2
        return data[:, :, cropx:-cropx, cropy:-cropy, :]



"""Plotting"""


def plot_patch(patch,
               axes=None,
               title='my_patch',
               output_size=None,
               alpha=1.0):

    if output_size:
        patch = fit_data(patch, output_size)

    if axes is None:
        _, ax = plt.subplots(1, 1)
    else:
        ax = axes
    ax.imshow(patch, alpha=alpha)
    if axes is None:
        plt.show()


def plot_mask(mask,
              label_map,
              labels,
              color_map,
              axes=None,
              title='',
              plot_legend=False,
              output_size=None,
              alpha=1.0):

    if output_size:
        print('len', len(mask))
        mask = fit_data(mask, output_size)

    color_labels = {label_map[label] - 1: (color, labels[label_map[label]]) for label, color in color_map.items()}
    if alpha < 1.0:
        color_labels.update({-1: ('white', 'NA')})
    else:
        color_labels.update({-1: ('black', 'NA')})
    cmap = colors.ListedColormap([color_labels[index][0]
                                  for index in sorted(map(int, list(color_labels.keys())))])

    lgd = None
    if axes is None:
        _, ax = plt.subplots(1, 1)
    else:
        ax = axes

    color_indexes = list(color_labels.keys())
    ax.imshow(mask, interpolation='nearest', cmap=cmap,
              vmin=min(color_indexes), vmax=max(color_indexes), alpha=alpha)
    ax.set_title(title)
    if plot_legend:
        patches = [mpatches.Patch(color=color_labels[index][0], label="Label: {l}".format(
            l=color_labels[index][1])) for index in sorted(map(int, list(color_labels.keys())))]
        # put those patched as legend-handles into the legend
        lgd = ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if axes is None:
        plt.show()


# def plot_one_hot(mask, output_shape):
#     fig, axes = plt.subplots(nrows=1, ncols=mask.shape[-1], figsize=(15, 5))
#     label_idx = 0
#     for ax in axes:
#         im = ax.imshow(mask[..., label_idx].reshape(output_shape[0], output_shape[1]))
#         label_idx += 1

#     fig.colorbar(im, ax=axes.ravel().tolist())
#     plt.show()


# def plot_annotations(annotations, axes=None):
#     for annotation in annotations:
#         if axes == None:
#             if annotation.type == 'Point':
#                 plt.scatter(*annotation.coordinates())
#             else:
#                 plt.plot(*list(zip(*annotation.coordinates())))
#             plt.gca().invert_yaxis()
#             plt.axis('equal')
#             plt.show()
#         else:
#             if annotation.type == 'Point':
#                 axes.scatter(*annotation.coordinates())
#             else:
#                 axes.plot((*list(zip(*annotation.coordinates()))))


def load_data_source(yaml_path, source='server', modes=['training', 'validation']):
    with open(yaml_path) as f:
        data = yaml.load(f, yaml.FullLoader)

    # input image/xml/label paths
    data_source = {mode: [{source_key: os.path.join(source_path, data_item[source_key])
                           for source_key, source_path in data['sources'][source].items()}
                          for data_item in data['data'][mode]]
                   for mode in modes}

    return data_source


def find_pairs(image_paths, annotation_paths, exclude='mask'):
    
    pairs = []
    for image_path in image_paths:
        if exclude in image_path:
            continue
        for annotation_path in annotation_paths:
            if exclude in annotation_path:
                continue
            if os.path.splitext(os.path.basename(image_path))[0] in annotation_path:
                pair = {'image_path': image_path, 'annotation_path': annotation_path}
                pairs.append(pair)
    return pairs


def create_data_source(data_folder,
                       annotations_path=None,
                       images_extension='.tif',
                       annotations_extension='.xml',
                       mode='training',
                       exclude='mask'):
    """
    Function to create image,label pairs

    Parameters
    ----------
    data_folder : string
        String of the path where the Whole Slide Images are located.
    annotations_path : string, optional
        String of the path where the Annotations are located. The default is None.
    images_extension : string, optional
        extension of the Whole Slide Images. The default is '.tif'.
    annotations_extension : string, optional
        extension of the Whole Slide Images Extensions. The default is '.xml'.
    mode : string, optional
        mode of the data_source. The default is 'training'.
    exclude : string, optional
        String pattern to exclude in the getting of the images. The default is 'mask'.

    Returns
    -------
    dict
        dict with (image,label) pairs.

    """

    if annotations_path is None:
        annotations_path = data_folder
    image_paths = glob.glob(os.path.join(data_folder, '*'+images_extension))
    annotation_paths = glob.glob(os.path.join(annotations_path, '*'+annotations_extension))
    pairs = find_pairs(image_paths, annotation_paths, exclude)
    return {mode: pairs}


def convert_image_annotation_to_xml(image_annotation, out_path, label_map, color_map):
    def write_text(path, text):
        with open(str(path), 'w') as the_file:
            the_file.write(text)

    def prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    top = Element('ASAP_Annotations')
    groups = SubElement(top, 'AnnotationGroups')
    for lab in label_map:
        groupel = SubElement(groups, 'Group', attrib=dict(Name=lab, PartOfGroup='None', Color=color_map[lab]))

    annosel = SubElement(top, 'Annotations')
    for i, p in enumerate(image_annotation.annotations):

        annoel = SubElement(annosel, 'Annotation', attrib=dict(Name=f'Annotation {i}', Type='Polygon',
                                                               PartOfGroup=p.label_name,
                                                               Color=color_map[p.label_name]))
        coordsel = SubElement(annoel, 'Coordinates')
        coords = p._coordinates
        for i_, coord in enumerate(coords):
            coordel = SubElement(coordsel, 'Coordinate', attrib=dict(Order=str(i_), X=str(coord[0]), Y=str(coord[1])))
    # print(prettify(top))
    xml_string = prettify(top)
    write_text(out_path, xml_string)
    # ElementTree(top).write(out_path)
    print('%s saved' % out_path)