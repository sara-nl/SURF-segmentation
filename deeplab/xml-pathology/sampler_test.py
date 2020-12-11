import sys
sys.path.insert(0,'/home/rubenh/xml-pathology/xmlpathology')
print(sys.path)

from xmlpathology.batchgenerator.utils import create_data_source
from xmlpathology.batchgenerator.generators import BatchGenerator
from xmlpathology.batchgenerator.core.samplers import LabelSamplerLoader, SamplerLoader
from xmlpathology.batchgenerator.core.samplers import SegmentationLabelSampler, Sampler
from xmlpathology.batchgenerator.callbacks import OneHotEncoding, FitData
import pdb
import time
# parser = ArgumentConfigParser('./parameters.yml', description='HookNet')
# config = parser.parse_args()

datasource_train = create_data_source(data_folder='/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor',
                                      annotations_path='/nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/XML',
                                      images_extension='.tif',
                                      annotations_extension='.xml',
                                      mode='training')

datasource_validation = create_data_source(data_folder='/nfs/managed_datasets/CAMELYON16/Testset/Images',
                                           annotations_path='/nfs/managed_datasets/CAMELYON16/Testset/Ground_Truth/Annotations',
                                           images_extension='.tif',
                                           annotations_extension='.xml',
                                           mode='validation')

label_map = {'_0': 1, '_2': 2}

config = dict()
config['batch_size']=1
config['input_shape'] = [8192,8192,3]
config['resolutions'] = [0.25]
config['cpus'] = 8

# initialize batchgenerator
t1 = time.time()
batchgen_train = BatchGenerator(data_sources=datasource_train,
                                label_map=label_map,
                                batch_size=config['batch_size'],
                                cpus=config['cpus'],
                                sampler_loader=SamplerLoader(class_=Sampler, input_shapes=[config['input_shape']],
                                                             spacings=config['resolutions']),
                                label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                                log_path='./logs/',
                                sample_callbacks=[OneHotEncoding()])
print(f'Starting Batch Generator took {time.time() - t1} seconds with {config["cpus"]} cpus')

batchgen_validation = BatchGenerator(data_sources=datasource_validation,
                                     label_map=label_map,
                                     batch_size=config['batch_size'],
                                     sampler_loader=SamplerLoader(class_=Sampler,
                                                                  input_shapes=[config['input_shape']],
                                                                  spacings=config['resolutions']),
                                     label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                                     log_path='./logs/',
                                     sample_callbacks=[OneHotEncoding()])


batchgen_train.start()
batchgen_validation.start()
batch = batchgen_train.batch('training')


# imports
# from pprint import pprint
# from matplotlib import pyplot as plt
# import yaml
# import os

# from xmlpathology.argconfigparser import argconfigparser

# from xmlpathology.xmlbatchgenerator.core.generators import XmlBatchGeneratorVanilla
# from xmlpathology.io.dataset import DataSet, DataSetLoader
# from xmlpathology.io.annotationparser import AnnotationParserLoader, AsapAnnotationParser
# from xmlpathology.xmlbatchgenerator.callbacks.samplecallbacks.fityolo import FitYolo
# import time
# import numpy as np
# import glob

# import multiprocessing

# multiprocessing.set_start_method('spawn', force=True)


# if __name__ == '__main__':
#     """

#     Setup

#     """

#     """
#     Data setup
#     """

#     # get the data from datasource.yml:

#     # ---
#     # training:
#     #   -
#     #     annotation_path: /home/mart/radboudumc/data/xmls/level1/T10-00485-I-19-1-gr3.xml
#     #     image_path: /home/mart/radboudumc/data/images/level1/T10-00485-I-19-1-gr3.tif
    

#     data_sources_path = '/home/mart/radboudumc/lib/data.yml'
#     with open(data_sources_path) as f:
#         data_sources = yaml.load(f, yaml.FullLoader)

#     print('data sources:')
#     pprint(data_sources)
#     """
#     Label map
#     """

#     # # set the label_map, which maps labels that are in the annotation file to label_values:
#     # label_map = \
#     # {
#     #     'dcis': 1,
#     #     'idc': 2,
#     #     'ilc': 3,
#     #     'fatty tissue': 4,
#     #     'stroma': 5,
#     #     'erythrocytes': 6,
#     #     'non malignant epithelium': 7,
#     #     'inflammatory cells': 8,
#     #     'skin/nipple': 9
#     # }

#     label_map = \
#     {
#         'polygon': 0,
#         'dot': 1,
#     }

#     inverse_label_map = {value:key for key, value in label_map.items()}

#     print('label map:')
#     pprint(label_map)


#     fit_yolo_callback = FitYolo(label_map, (128,128,3), (16,16), 1, 24)
#     # lets first define all the stuff that we need to initialize the XmLBatchGenerator:

#     """
#     Starting the XmlBatchGeneratorVanilla
#     """

#     batchgenerator = XmlBatchGeneratorVanilla(data_sources=data_sources,
#                                             label_map=label_map,
#                                             open_images_ahead=True,
#                                             batch_size=16,
#                                             input_shape=(128,128,3),
#                                             cpus=4,
#                                             strict_point_sampling=True,
#                                             sample_callbacks=[fit_yolo_callback],
#                                             log_path='/home/rubenh/logfiles/')

#     batchgenerator.start()

#     b = batchgenerator.batch('training')

#     print('hoi')