import numpy as np
import pandas as pd
from vectorPreProcess import vectorsPreProcces as vpp

class PearsonC:

    def __init__(self, is_objects: bool = True, z_levels: list = list([1, 2, 3, 4]), intensity_feature_name: str = "IntDen"):
        self.is_objects = is_objects
        self.z_levels = z_levels
        self.intensity_feature_name = intensity_feature_name
        self.vpp = vpp()

    def calc_pearson(self, v1:pd.DataFrame, v2:pd.DataFrame, split_z_axis: bool = True):
        v1, v2 = self.vpp.same_length_vectors(v1.loc[:, ['X', 'Y', 'Z', self.intensity_feature_name]],
                                              v2.loc[:, ['X', 'Y', 'Z', self.intensity_feature_name]])
        if self.is_objects and split_z_axis:
            coeff_for_z_level = list()
            for i in range(0, len(self.z_levels) - 1):
                v1_trimmed = v1.loc[(v1['Z'] >= self.z_levels[i]) * (v1['Z'] < self.z_levels[i + 1])]
                v2_trimmed = v2.loc[(v2['Z'] >= self.z_levels[i]) * (v2['Z'] < self.z_levels[i + 1])]
                v1_trimmed, v2_trimmed = self.vpp.same_length_vectors(v1_trimmed, v2_trimmed)
                coeff_for_z_level.append(self.calc_pearson_objects(v1_trimmed, v2_trimmed))
            return coeff_for_z_level
        elif self.is_objects:
            return self.calc_pearson_objects(v1, v2)
        else:
            return self.calc_pearson_pixels(v1, v2)

    def calc_pearson_objects(self, v1, v2):
        v1_objects_intensities = v1[self.intensity_feature_name].values
        v2_objects_intensities = v2[self.intensity_feature_name].values
        total_average_v1_pixels = np.average(v1_objects_intensities)
        total_average_v2_pixels = np.average(v2_objects_intensities)

        im1_pixels_intensities_average_subtracted = v1_objects_intensities - total_average_v1_pixels
        im2_pixels_intensities_average_subtracted = v2_objects_intensities - total_average_v2_pixels

        nom = np.sum(im1_pixels_intensities_average_subtracted * im2_pixels_intensities_average_subtracted)
        denom = np.sqrt(np.sum(np.power(im1_pixels_intensities_average_subtracted, 2, dtype="float"),
                               dtype="float")) * np.sqrt(
            np.sum(np.power(im2_pixels_intensities_average_subtracted, 2, dtype="float"), dtype="float"))
        return nom / denom

    def calc_pearson_pixels(self, im1, im2, channels_to_consider:tuple =(1,2)):
        im1_pixels_intensities = im1[:,:, channels_to_consider[0]].flatten()
        im2_pixels_intensities = im2[:, :, channels_to_consider[1]].flatten()
        total_average_im1_pixels = np.average(im1_pixels_intensities)
        total_average_im2_pixels = np.average(im2_pixels_intensities)

        im1_pixels_intensities_average_subtracted = im1_pixels_intensities - total_average_im1_pixels
        im2_pixels_intensities_average_subtracted = im2_pixels_intensities - total_average_im2_pixels

        nom = np.sum(im1_pixels_intensities_average_subtracted * im2_pixels_intensities_average_subtracted)
        denom = np.sqrt(np.sum(np.power(im1_pixels_intensities_average_subtracted, 2, dtype="float"),
                               dtype="float")) * np.sqrt(
            np.sum(np.power(im2_pixels_intensities_average_subtracted, 2, dtype="float"), dtype="float"))
        return nom / denom