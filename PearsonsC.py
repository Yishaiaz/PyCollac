import os

import numpy as np
import pandas as pd
from vectorPreProcess import vectorsPreProcces as vpp
import matplotlib.pyplot as plt
import seaborn as sns


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
                curr_coeff = 0 if np.sum(v1_trimmed.values)==0 or np.sum(v2_trimmed.values)==0 else self.calc_pearson_objects(v1_trimmed, v2_trimmed)
                coeff_for_z_level.append(curr_coeff)

            last_z_v1_trimmed, last_z_v2_trimmed = self.vpp.same_length_vectors(v1.loc[(v1['Z'] >= self.z_levels[len(z_levels)-1])],
                                                                                v2.loc[(v2['Z'] >= self.z_levels[len(z_levels)-1])])
            curr_coeff = 0 if np.sum(last_z_v1_trimmed.values) == 0 or np.sum(last_z_v2_trimmed.values) == 0 else self.calc_pearson_objects(last_z_v1_trimmed, last_z_v2_trimmed)
            coeff_for_z_level.append(curr_coeff)
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

        im2_pixels_intensities_average_subtracted = v2_objects_intensities - total_average_v2_pixels
        im1_pixels_intensities_average_subtracted = v1_objects_intensities - total_average_v1_pixels

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

sns.set()

pearsonForEachZForEachFile = pd.DataFrame(columns=['file_name', 'Z=1', 'Z=2', 'Z=3', 'Z=4', 'Z=5'])
MAIN_DIR_NAME = "/Users/yishaiazabary/Desktop/University/DNA Deformat proteins research/extractedStatistics/"
# MAIN_DIR_NAME = "/Users/yishaiazabary/PycharmProjects/PyCollac/Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)"
path_to_all_channels_dirs = MAIN_DIR_NAME
channels_dirs = [dir_name[0]+"/" for dir_name in os.walk(path_to_all_channels_dirs)][1:]
channel1FilesList = [file_path[2] for file_path in os.walk(channels_dirs[0])][0]
channel1FilesList.remove('.DS_Store')
channel2FilesLisr = [file_path[2] for file_path in os.walk(channels_dirs[1])][0]
channel2FilesLisr.remove('.DS_Store')
for file_ctr in range(len(channel1FilesList)):
    # Green to red
    first_channel_objects = pd.read_csv(channels_dirs[0]+channel1FilesList[file_ctr]).loc[:, ['X', 'Y', 'Z', 'IntDen']]
    z_levels = first_channel_objects['Z'].astype('int').unique()
    second_channel_objects = pd.read_csv(channels_dirs[1]+channel2FilesLisr[file_ctr]).loc[:, ['X', 'Y', 'Z', 'IntDen']]
    # Red to Green
    # first_channel_objects = pd.read_csv(channels_dirs[0] + channel2FilesLisr[file_ctr]).loc[:,
    #                         ['X', 'Y', 'Z', 'IntDen']]
    # z_levels = first_channel_objects['Z'].astype('int').unique()
    # second_channel_objects = pd.read_csv(channels_dirs[1] + channel1FilesList[file_ctr]).loc[:,
    #                          ['X', 'Y', 'Z', 'IntDen']]

    pc = PearsonC(is_objects=True, z_levels=np.sort(z_levels), intensity_feature_name="IntDen")
    print("analyzed: "+channel1FilesList[file_ctr])
    res = pc.calc_pearson(first_channel_objects, second_channel_objects, True)
    while len(res) < 5:
        res += [0]
    file_title = channel1FilesList[file_ctr][:-8].split('_')
    file_title = file_title[0].split(' ')[4] + "_"+file_title[2] + "_" + file_title[3]
    pearsonForEachZForEachFile.loc[file_ctr] = [file_title]+res

pearsonForEachZForEachFile = pearsonForEachZForEachFile.set_index('file_name', drop=True)
pearsonForEachZForEachFile = pearsonForEachZForEachFile.astype(float)
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(pearsonForEachZForEachFile, annot=True, ax=ax)
ax.set_title = "Pearson RedToGreen"

plt.tight_layout()
plt.show()
# f.savefig("RedToGreen.png", dpi=200)