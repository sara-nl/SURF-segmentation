import glob
import logging
from typing import Optional, Dict
import platform
import os
from datetime import datetime
import xml.etree.cElementTree as ET
from xml.dom import minidom
import json

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pathlib



def remove_absl_logging_handler():
    """
    Logging issue with tensorflow (absl.logging)

    see:
    https://github.com/abseil/abseil-py/issues/99
    https://github.com/abseil/abseil-py/issues/102

    """
    try:
        import absl.logging
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
    except Exception:
        pass

def save_dict2json(json_path, dict_md5):
    with open(json_path, 'w') as fp:
        json.dump(dict_md5, fp, indent=4)


def system_status():
    osys, name, version, _, _, _ = platform.uname()
    version = version.split('-')[0]
    cores = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory()[2]
    disk_percent = psutil.disk_usage('/')[3]
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    running_since = boot_time.strftime("%A %d. %B %Y")
    response = "\nRunning on %s version %s.  " % (osys, version)
    response += "\nThis system is named %s and has %s CPU cores.  " % (name, cores)
    response += "\nCurrent disk_percent is %s percent.  " % disk_percent
    response += "\nCurrent CPU utilization is %s percent.  " % cpu_percent
    response += "\nCurrent memory utilization is %s percent. " % memory_percent
    response += "\nit's running since %s." % running_since
    response += "\n"
    return response


class Logger:
    """
    Logger class

    """

    def __init__(self, datasets: Optional[Dict] = None, log_path: Optional[str] = None, timestamp=True):

        remove_absl_logging_handler()
        self._log_path = log_path
        self._datasets = datasets
        self._sample_info = []
        self._set = False


        if self._log_path:
            if timestamp:
                self._log_path = os.path.join(self._log_path, str(datetime.timestamp(datetime.now())).replace('.', ''))
            pathlib.Path(self._log_path).mkdir(parents=True, exist_ok=True)
            self._set = True



    def get_logger(self, name: str) -> logging.Logger:

        if not self._set:
            return logging.getLogger('null')

        # create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create handler and set level to debug
        log_filename = os.path.join(self._log_path, name) + '.log'
        ch = logging.FileHandler(log_filename, mode='w')
        ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)
        return logger


    def update(self, auxiliaries):
        for sampler_aux in auxiliaries['sampler']:
            for info in sampler_aux['sample_info']:
                self._sample_info.append(info)


    def finalize(self):
        if self._set:
            self._merge_sampler_logs()
            self._save_figure_samples()
            self._save_sample_info()

    def _save_sample_info(self):
        save_dict2json(os.path.join(self._log_path, 'sample_info.json'), self._sample_info)

    def _merge_sampler_logs(self):
        logs = glob.glob(os.path.join(self._log_path, '*.log'))
        logs = [log for log in logs if 'Sampler' in log]
        lines = []
        for log in logs:
            with open(log) as file:
                for line in file.readlines():
                    lines.append(line)
        lines.sort(key=lambda lines: lines.split()[:2])

        with open(os.path.join(self._log_path, 'sampler.log'), 'w') as file:
            for line in lines:
                file.write("%s" % line)
        for log in logs:
            os.remove(log)


    def _save_figure_samples(self):
        if self._datasets:
            plt.ioff()

            path_to_log_figures = os.path.join(self._log_path, 'sample_figures')
            if path_to_log_figures and not os.path.exists(path_to_log_figures):
                os.mkdir(path_to_log_figures)

            # visualize sampled center coordinates in image annotations
            samples = {}
            for sample in self._sample_info:
                samples.setdefault(sample['image_annotation_index'], []).append(sample)

            for image_annotation_index, image_samples in samples.items():
                points = []
                rects = []
                mode = None
                for sample in image_samples:
                    mode = sample['mode']
                    annotation_index = sample['annotation_index']
                    center_x, center_y = sample['center']
                    width, height, _ = sample['patch_shape']
                    ratio = self._datasets[sample['mode']].image_annotations[image_annotation_index].get_ratio(sample['pixel_spacing'])

                    annotation = self._datasets[sample['mode']].image_annotations[image_annotation_index].annotations[annotation_index]
                    points.append((center_x, center_y))
                    if annotation.type == 'polygon':
                        rects.append(annotation.coordinates())
                    x1 = center_x - (width*ratio)//2
                    y1 = center_y - (height*ratio)//2
                    x2 = center_x + (width*ratio)//2
                    y2 = center_y + (height*ratio)//2
                    rects.append([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])

                if mode:
                    annotation_log_path = os.path.join(path_to_log_figures, os.path.basename(self._datasets[mode].image_annotations[image_annotation_index].annotation_path))
                    ext = os.path.splitext(annotation_log_path)[-1]
                    self.write_xml(annotation_log_path.replace(ext, '_samples.xml'), points=points, rois=rects)


    def write_xml(self, out_path, points=None, labels=[], colors=[], rois=None, roi_labels=[], roi_colors=[]):
        """
        """
        # check point or rois
        if not points and not rois:
            raise ValueError('either points or rois should be set.')

        #
        if not isinstance(points, list) and points is not None:
            points = [points]
        if not isinstance(rois, list) and rois is not None:
            rois = [rois]

        # the root of the xml file.
        root = ET.Element("ASAP_Annotations")

        # writing each anno one by one.
        annos = ET.SubElement(root, "Annotations")

        # set labels
        if not labels:
            labels = ["ROI"]*len(points)
        if not roi_labels:
            roi_labels = ["ROI"]*len(rois)

        # set colors
        if not colors:
            colors = ["#000000"]*len(points)
        if not roi_colors:
            roi_colors = ["#000000"]*len(rois)

        # writing for the rectangular ROIs
        if rois:
            for idx0, rect in enumerate(rois):
                anno = ET.SubElement(annos, "Annotation")
                anno.set("Name", "Annotation "+str(idx0))
                anno.set("Type", "Polygon")
                anno.set("PartOfGroup", roi_labels[idx0])
                anno.set("Color", roi_colors[idx0])

                coords = ET.SubElement(anno, "Coordinates")
                for ridx, r in enumerate(rect):
                    coord = ET.SubElement(coords, "Coordinate")
                    coord.set("Order", str(ridx))
                    coord.set("X", str(r[0]))
                    coord.set("Y", str(r[1]))

        # writing for the dot annots
        if points:
            for idx, point in enumerate(points):
                lbl = labels[idx]
                clr = colors[idx]

                anno = ET.SubElement(annos, "Annotation")
                anno.set("Name", "Annotation "+str(idx+len(rois)))
                anno.set("Type", "Dot")
                anno.set("PartOfGroup", lbl)
                anno.set("Color", clr)

                coords = ET.SubElement(anno, "Coordinates")
                coord = ET.SubElement(coords, "Coordinate")
                coord.set("Order", "0")
                coord.set("X", str(point[0]))
                coord.set("Y", str(point[1]))

        # writing the last groups part
        anno_groups = ET.SubElement(root, "AnnotationGroups")

        # get the group names and colors from the annotations.
        # annotation labels and roi labels
        full_labels = labels+roi_labels
        # annotatoin colors and roi colors
        full_colors = colors+roi_colors
        # make the set of the labels and the colors
        labelset = list(np.unique(np.array(full_labels)))
        colorset = [full_colors[full_labels.index(l)] for l in labelset]

        for label, color in zip(labelset, colorset):
            group = ET.SubElement(anno_groups, "Group")
            group.set("Name", label)
            group.set("PartOfGroup", "None")
            group.set("Color", color)
            attr = ET.SubElement(group, "Attributes")

        # writing to the xml file with indentation
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        with open(out_path, "w") as f:
            f.write(xmlstr)
