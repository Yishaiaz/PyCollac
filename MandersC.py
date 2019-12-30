import numpy as np
import pandas as pd
from vectorPreProcess import vectorsPreProcces as vpp
import matplotlib.pyplot as plt
import seaborn as sns
import os


class MandersC:

    def __init__(self, is_objects: bool = True, z_levels: list = list([1, 2, 3, 4]), intensity_feature_name: str = "IntDen"):
        self.is_objects = is_objects
        self.z_levels = z_levels
        self.intensity_feature_name = intensity_feature_name
        self.vpp = vpp()

    def calc_m_for_single_file(self, v1: pd.DataFrame, v2: pd.DataFrame, split_z_axis: bool = True):
        v1, v2 = self.vpp.same_length_vectors(v1.loc[:, ['X', 'Y', 'Z', self.intensity_feature_name]],
                                         v2.loc[:, ['X', 'Y', 'Z', self.intensity_feature_name]])
        if split_z_axis:
            coeff_for_z_level = list()
            if self.is_objects:
                mc = self.MandersCoeffForObjects(self.intensity_feature_name)
                for i in range(0, len(self.z_levels) - 1):
                    v1_trimmed = v1.loc[(v1['Z'] >= self.z_levels[i]) * (v1['Z'] < self.z_levels[i + 1])]
                    v2_trimmed = v2.loc[(v2['Z'] >= self.z_levels[i]) * (v2['Z'] < self.z_levels[i + 1])]
                    v1_trimmed, v2_trimmed = self.vpp.same_length_vectors(v1_trimmed, v2_trimmed)
                    coeff_for_z_level.append(mc.calc_m(v1_trimmed, v2_trimmed))
                return coeff_for_z_level
            else:
                mc = self.MandersCoeffForPixels()
                return mc.calc_m(v1, v2, channels_to_consider=(1, 2))
        else:
            if self.is_objects:
                mc = self.MandersCoeffForObjects(self.intensity_feature_name)
                return mc.calc_m(v1, v2)
            else:
                mc = self.MandersCoeffForPixels()
                return mc.calc_m(v1, v2, channels_to_consider=(1, 2))

    def calc_MOC_for_single_file(self, v1: pd.DataFrame, v2: pd.DataFrame):
        v1, v2 = self.vpp.same_length_vectors(v1.loc[:, ['X', 'Y', 'Z', self.intensity_feature_name]],
                                         v2.loc[:, ['X', 'Y', 'Z', self.intensity_feature_name]])
        coeff_for_z_level = list()
        if self.is_objects:
            mc = self.MandersCoeffForObjects(self.intensity_feature_name)
            for i in range(0, len(self.z_levels) - 1):
                v1_trimmed = v1.loc[(v1['Z'] >= self.z_levels[i]) * (v1['Z'] < self.z_levels[i + 1])]
                v2_trimmed = v2.loc[(v2['Z'] >= self.z_levels[i]) * (v2['Z'] < self.z_levels[i + 1])]
                v1_trimmed, v2_trimmed = self.vpp.same_length_vectors(v1_trimmed, v2_trimmed)
                coeff_for_z_level.append(mc.calc_MOC(v1_trimmed, v2_trimmed))
            return coeff_for_z_level
        else:
            mc = self.MandersCoeffForPixels()
            return mc.calc_MOC(v1, v2, channels_to_consider=(1, 2))

    class MandersCoeffForPixels:

        def calc_m(self, im1, im2, channels_to_consider: tuple = (1, 2)):
            total_Xi_denom = 0
            total_Xi_nom = 0

            for i in range(0, len(im1)):
                for j in range(0, len(im1[i])):
                    im1_avg = im1[i, j, channels_to_consider[0]]
                    im2_avg = im2[i, j, channels_to_consider[1]]
                    total_Xi_denom += im1_avg
                    total_Xi_nom += 0 if im2_avg < 20 else im1_avg
            return total_Xi_nom / total_Xi_denom

        def calc_MOC(self, im1, im2, channels_to_consider: tuple = (1, 2)):
            im1 = im1[:, :, channels_to_consider[0]].flatten()
            im2 = im2[:, :, channels_to_consider[1]].flatten()
            total_nom = np.sum(im1 * im2, dtype="float")
            total_denom = np.sqrt(np.sum(np.power(im1, 2, dtype="float"))) * np.sqrt(
                np.sum(np.power(im2, 2, dtype="float")))
            return total_nom / total_denom

    class MandersCoeffForObjects:
        def __init__(self, intensity_feature_name: str):
            self.intensity_feature_name = intensity_feature_name

        def calc_m(self, main_spots:pd.DataFrame, secondary_spots:pd.DataFrame):
            return np.sum(main_spots[(secondary_spots[['IntDen']] > 0).values][self.intensity_feature_name])\
                    / np.sum(main_spots[self.intensity_feature_name])

        def calc_MOC(self, v1, v2):
            v1 = v1[self.intensity_feature_name]
            v2 = v2[self.intensity_feature_name]

            total_nom = np.sum(v1.values * v2.values)
            total_denom = np.sqrt(np.sum(np.power(v1.values, 2, dtype="float"))) * np.sqrt(
                np.sum(np.power(v2.values, 2, dtype="float")))
            return total_nom / total_denom

sns.set()

mandersMocForEachZForEachFile = pd.DataFrame(columns=['file_name', 'Z=1', 'Z=2', 'Z=3', 'Z=4', 'Z=5'])
MAIN_DIR_NAME = "/Users/yishaiazabary/Desktop/University/DNA Deformat proteins research/extractedStatistics/"
# MAIN_DIR_NAME = "/Users/yishaiazabary/PycharmProjects/PyCollac/Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)"
path_to_all_channels_dirs = MAIN_DIR_NAME
channels_dirs = [dir_name[0]+"/" for dir_name in os.walk(path_to_all_channels_dirs)][1:]
channel1FilesList = [file_path[2] for file_path in os.walk(channels_dirs[0])][0]
channel1FilesList.remove('.DS_Store')
channel2FilesLisr = [file_path[2] for file_path in os.walk(channels_dirs[1])][0]
channel2FilesLisr.remove('.DS_Store')
for file_ctr in range(len(channel1FilesList)):
    # Green to Red
    first_channel_objects = pd.read_csv(channels_dirs[0]+channel1FilesList[file_ctr]).loc[:, ['X', 'Y', 'Z', 'IntDen']]
    z_levels = first_channel_objects['Z'].astype('int').unique()
    second_channel_objects = pd.read_csv(channels_dirs[1]+channel2FilesLisr[file_ctr]).loc[:, ['X', 'Y', 'Z', 'IntDen']]
    # Red to Green
    # first_channel_objects = pd.read_csv(channels_dirs[1] + channel2FilesLisr[file_ctr]).loc[:,
    #                         ['X', 'Y', 'Z', 'IntDen']]
    # z_levels = first_channel_objects['Z'].astype('int').unique()
    # second_channel_objects = pd.read_csv(channels_dirs[0] + channel1FilesList[file_ctr]).loc[:,
    #                          ['X', 'Y', 'Z', 'IntDen']]

    mc = MandersC(is_objects=True, z_levels=np.sort(z_levels), intensity_feature_name="IntDen")
    print("analyzed: "+channel1FilesList[file_ctr])
    res = mc.calc_MOC_for_single_file(first_channel_objects, second_channel_objects)
    # res = mc.calc_m_for_single_file(first_channel_objects, second_channel_objects, True)
    while len(res) < 5:
        res += [0]
    file_title = channel1FilesList[file_ctr][:-8].split('_')
    file_title = file_title[0].split(' ')[4] + "_"+file_title[2] + "_" + file_title[3]
    mandersMocForEachZForEachFile.loc[file_ctr] = [file_title] + res

mandersMocForEachZForEachFile = mandersMocForEachZForEachFile.set_index('file_name', drop=True)
mandersMocForEachZForEachFile = mandersMocForEachZForEachFile.astype(float)
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(mandersMocForEachZForEachFile, annot=True, ax=ax)
ax.set_title = "Manders RedToGreen"

plt.tight_layout()
plt.show()
dir = "InitialResults/MandersCoeff/MOC_CALCS/"
plot_file_name = "GreenToRedMoc"
f.savefig("{0}{1}.png".format(dir, plot_file_name), dpi=200)
