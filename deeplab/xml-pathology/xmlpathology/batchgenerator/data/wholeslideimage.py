import os
from abc import ABC, abstractmethod
from typing import List, Tuple
import pdb
import numpy as np
import time

try:
    from multiresolutionimageinterface import MultiResolutionImageReader, MultiResolutionImage
except ImportError:
    print('failed to import MultiResolutionImage')

try:
    from openslide import OpenSlide
except ImportError:
    print('failed to import OpenSLide')


def whole_slide_image_factory(backend='openslide'):
    if backend == 'openslide':
        return WholeSlideImageOpenSlide
    elif backend == 'asap':
        return WholeSlideImageASAP
    raise ValueError(f'unsupported backend {backend} for WholeSlideImage')


class InvalidSpacingError(ValueError):
    def __init__(self, image_path, spacing, spacings, margin):

        super().__init__(f"Image: '{image_path}\', with available pixels spacings: {spacings}, does not contain a level corresponding to a pixel spacing of {spacing} +- {margin}")

        self._image_path = image_path
        self._spacing = spacing
        self._spacings = spacings
        self._margin = margin

    def __reduce__(self):
        return (InvalidSpacingError, (self._image_path,
                                      self._spacing,
                                      self._spacings,
                                      self._margin))


class UnsupportedVendorError(KeyError):
    def __init__(self, image_path, properties):

        super().__init__(f"Image: '{image_path}\', with properties: {properties}, is not in part of the supported vendors")

        self._image_path = image_path
        self._properties = properties

    def __reduce__(self):
        return (UnsupportedVendorError, (self._image_path, self._properties))


class WholeSlideImage(ABC):

    def __init__(self, image_path: str, spacing_margin_ratio: float = 1/10) -> None:
        self._image_path = image_path
        self._extension = os.path.splitext(image_path)[-1]
        self._spacing_margin_ratio = spacing_margin_ratio
        self._shapes = self._init_shapes()
        self._downsamplings = self._init_downsamplings()
        self._spacings = self._init_spacings()

    @property
    def filepath(self) -> str:
        return self._image_path

    @property
    def extension(self) -> str:
        return self._extension

    @property
    def spacings(self) -> List[float]:
        return self._spacings

    @property
    def shapes(self) -> List[Tuple[int, int]]:
        return self._shapes

    @property
    def downsamplings(self) -> List[float]:
        return self._downsamplings

    @property
    def level_count(self) -> int:
        return len(self.spacings)

    def get_downsampling_from_level(self, level: int) -> float:
        return self.downsamplings[level]

    def get_level_from_spacing(self, spacing: float) -> int:
        spacing_margin = spacing*self._spacing_margin_ratio
        for level, spacing_ in enumerate(self.spacings):
            if abs(spacing_ - spacing) <= spacing_margin:
                return level
        raise InvalidSpacingError(self._image_path, spacing, self.spacings, spacing_margin)

    def get_downsampling_from_spacing(self, spacing: float) -> float:
        return self.get_downsampling_from_level(self.get_level_from_spacing(spacing))

    @abstractmethod
    def _init_shapes(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def _init_downsamplings(self) -> List[float]:
        pass

    @abstractmethod
    def _init_spacings(self) -> List[float]:
        pass

    @abstractmethod
    def get_patch(self,
                  x: int,
                  y: int,
                  width: int,
                  height: int,
                  spacing: float,
                  center: bool = True,
                  relative: bool = False) -> np.ndarray:
        pass


class WholeSlideImageOpenSlide(OpenSlide, WholeSlideImage):

    def __init__(self, image_path: str) -> None:
        OpenSlide.__init__(self, image_path)
        WholeSlideImage.__init__(self, image_path)

    def get_patch(self,
                  x: int,
                  y: int,
                  width: int,
                  height: int,
                  spacing: float,
                  center: bool = True,
                  relative: bool = False) -> np.ndarray:

        downsampling = int(self.get_downsampling_from_spacing(spacing))
        level = self.get_level_from_spacing(spacing)
        if relative:
            x, y = x*downsampling, y*downsampling
        if center:
            x, y = x-downsampling*(width//2), y-downsampling*(height//2)
        
        t1 = time.time()
        result = np.array(super().read_region((int(x), int(y)), int(level), (int(width), int(height))))[:,:,:3]
        # print(f"Returned array in {time.time() - t1} seconds")
        return result

    def _init_shapes(self) -> List[Tuple[int, int]]:
        return self.level_dimensions

    def _init_downsamplings(self) -> List[float]:
        return self.level_downsamples

    def _init_spacings(self) -> List[float]:
        spacing = None
        try:
            spacing = float(self.properties['openslide.mpp-x'])
        except KeyError as key_error:
            try:
                unit = {'cm': 10000, 'centimeter': 10000}[self.properties['tiff.ResolutionUnit']]
                res = float(self.properties['tiff.XResolution'])
                spacing = unit/res
            except KeyError as key_error:
                raise UnsupportedVendorError(self._image_path, self.properties) from key_error

        return [spacing*self.get_downsampling_from_level(level) for level in range(super().level_count)]


# class WholeSlideImageASAP(MultiResolutionImage, WholeSlideImage):
#     def __init__(self, image_path: str) -> None:
#         self.__dict__.update(MultiResolutionImageReader().open(image_path).__dict__)
#         WholeSlideImage.__init__(self, image_path)
#         self.setCacheSize(0)

#     def get_patch(self,
#                   x: int,
#                   y: int,
#                   width: int,
#                   height: int,
#                   spacing: float,
#                   center: bool = True,
#                   relative: bool = False) -> np.ndarray:

#         downsampling = int(self.get_downsampling_from_spacing(spacing))
#         level = self.get_level_from_spacing(spacing)

#         if relative:
#             x, y = x*downsampling, y*downsampling
#         if center:
#             x, y = x-downsampling*(width//2), y-downsampling*(height//2)

#         return np.array(super().getUCharPatch(int(x), int(y), int(width), int(height), int(level)))

#     def _init_shapes(self) -> List[Tuple]:
#         try:
#             return [tuple(self.getLevelDimensions(level)) for level in range(self.getNumberOfLevels())]
#         except:
#             raise ValueError('shape en level errors')

#     def _init_downsamplings(self) -> List[float]:
#         return [self.getLevelDownsample(level) for level in range(self.getNumberOfLevels())]

#     def _init_spacings(self) -> List[float]:
#         try:
#             return [self.getSpacing()[0] * downsampling for downsampling in self.downsamplings]
#         except:
#             raise InvalidSpacingError(self._image_path, 0, [], 0)
