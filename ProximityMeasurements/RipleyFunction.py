import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import seaborn as sns
# from PearsonsC import PearsonC


class RipleyFunction:

    def __init__(self, spots_typeA: pd.DataFrame, spots_typeB: pd.DataFrame):
        self.spots_typeA = spots_typeA
        self.spots_typeB = spots_typeB
        self.plot_x_len = max(np.max(spots_typeA['X']), np.max(spots_typeB['X']))
        self.plot_y_len = max(np.max(spots_typeA['Y']), np.max(spots_typeB['Y']))

    def count_neighbors_in_radius(self, center_spot: pd.DataFrame, radius: float, x_df_title: str = "X", y_df_title: str = "Y", z_df_title: str = "Z"):
        distances_between_center_and_all_typeB = None
        if len(center_spot.values[0])== 2:
            # 2D
            x_sqrd_distances = np.power(center_spot[[x_df_title]].values[0][0] - self.spots_typeB[[x_df_title]], 2,
                                        dtype="float")
            y_sqrd_distances = np.power(center_spot[[x_df_title]].values[0][0] - self.spots_typeB[[y_df_title]], 2,
                                        dtype="float")
            distances_between_center_and_all_typeB = np.sqrt(y_sqrd_distances._values.flatten() + x_sqrd_distances._values.flatten())

        if len(center_spot.values[0]) == 3:
            # 3D
            x_sqrd_distances = np.power(center_spot[[x_df_title]].values[0][0] - self.spots_typeB[[x_df_title]], 2, dtype="float")
            y_sqrd_distances = np.power(center_spot[[x_df_title]].values[0][0] - self.spots_typeB[[y_df_title]], 2, dtype="float")
            z_sqrd_distances = np.power(center_spot[[x_df_title]].values[0][0] - self.spots_typeB[[z_df_title]], 2, dtype="float")
            distances_between_center_and_all_typeB = np.sqrt(y_sqrd_distances._values.flatten() + x_sqrd_distances._values.flatten() + z_sqrd_distances._values.flatten())

        return distances_between_center_and_all_typeB, len(np.where(distances_between_center_and_all_typeB<radius)[0])

    def calc_neighbors_in_radius_for_all_spots(self, radius):
        number_of_typeB_spots_in_radius_for_all_spots = list()
        average_of_spots_amount = 0
        for i in range(1, len(self.spots_typeA)+1):
            center_spot = self.spots_typeA.iloc[i-1:i]
            typeB_spots_distances_in_radius_for_spot, number_of_typeB_spots_in_radius_for_spot = self.count_neighbors_in_radius(center_spot, radius)
            spot_x = center_spot['X'].data.obj[0]
            spot_y = center_spot['Y'].data.obj[0]
            e_x = min(spot_x, self.plot_x_len - spot_x)
            e_y = min(spot_y, self.plot_x_len - spot_y)

            w = 1
            # for dist_between_spots in typeB_spots_distances_in_radius_for_spot:
            #     if radius > e_x and radius > e_y:
            #         w += (1 - (2 * math.acos(e_x / dist_between_spots) + 2 * math.acos(e_y / dist_between_spots)) * 2 * math.pi)
            #     elif radius > e_x:
            #         w += (1 - math.acos(e_x / dist_between_spots) / math.pi)
            #     elif radius > e_y:
            #         w += (1 - (math.acos(e_y / dist_between_spots) / math.pi))
            number_of_typeB_spots_in_radius_for_all_spots.append({
                "spot_number": i,
                "number_of_spots_in_radius": number_of_typeB_spots_in_radius_for_spot,
                "W_inver_times_amount_in_radius": w if w != 1 else 0,
                "weighted_distance": (1 / w) * number_of_typeB_spots_in_radius_for_spot
            })
            average_of_spots_amount += number_of_typeB_spots_in_radius_for_spot

        average_of_spots_amount = average_of_spots_amount/len(self.spots_typeA)
        return average_of_spots_amount, number_of_typeB_spots_in_radius_for_all_spots

    def calc_neighbors_in_radii(self, radii):
        results_for_radii = list()
        for i, radius in enumerate(radii):
            average, number_of_typeB_spots_in_radius_for_all_spots= self.calc_neighbors_in_radius_for_all_spots(radius)
            results_for_radii.append({
                "radius": radius,
                "neighbors_average_in_radius": average,
                "k_for_radius": math.pow(self.spots_typeB.shape[0], -2) * #math.pow(self.spots_typeB.shape[0], -2) if self.spots_typeB.shape[0] != 0 else 0 *
                                2 * math.pi * radius *
                                np.sum(np.array([x.get('weighted_distance') for x in number_of_typeB_spots_in_radius_for_all_spots]))

            })
        return results_for_radii

    def calc_neighbors_in_radii_df(self, radii):
        res_as_dictionaries = self.calc_neighbors_in_radii(radii=radii)
        radius = np.array([x.get('radius') for x in res_as_dictionaries])
        k = np.array([x.get('k_for_radius') for x in res_as_dictionaries])
        return pd.DataFrame({'Radius': radius, "K": k})


sns.set()

# pearsonForEachZForEachFile = pd.DataFrame(columns=['file_name', 'Z=1', 'Z=2', 'Z=3', 'Z=4', 'Z=5'])
# MAIN_DIR_NAME = "/Users/yishaiazabary/Desktop/University/DNA Deformat proteins research/extractedStatistics/"
# # MAIN_DIR_NAME = "/Users/yishaiazabary/PycharmProjects/PyCollac/Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)"
# path_to_all_channels_dirs = MAIN_DIR_NAME
# channels_dirs = [dir_name[0]+"/" for dir_name in os.walk(path_to_all_channels_dirs)][1:]
# channel1FilesList = [file_path[2] for file_path in os.walk(channels_dirs[0])][0]
# channel1FilesList.remove('.DS_Store')
# channel2FilesLisr = [file_path[2] for file_path in os.walk(channels_dirs[1])][0]
# channel2FilesLisr.remove('.DS_Store')
# for file_ctr in range(len(channel1FilesList)):
#     # Green to red
#     first_channel_objects = pd.read_csv(channels_dirs[0]+channel1FilesList[file_ctr]).loc[:, ['X', 'Y']]
#     z_levels = first_channel_objects['Z'].astype('int').unique()
#     second_channel_objects = pd.read_csv(channels_dirs[1]+channel2FilesLisr[file_ctr]).loc[:, ['X', 'Y']]
#     # Red to Green
#     # first_channel_objects = pd.read_csv(channels_dirs[0] + channel2FilesLisr[file_ctr]).loc[:,
#     #                         ['X', 'Y', 'Z', 'IntDen']]
#     # z_levels = first_channel_objects['Z'].astype('int').unique()
#     # second_channel_objects = pd.read_csv(channels_dirs[1] + channel1FilesList[file_ctr]).loc[:,
#     #                          ['X', 'Y', 'Z', 'IntDen']]
#
#     pc = PearsonC(is_objects=True, z_levels=np.sort(z_levels))
#     print("analyzed: "+channel1FilesList[file_ctr])
#     res = pc.calc_pearson(first_channel_objects, second_channel_objects, True)
#     while len(res) < 5:
#         res += [0]
#     file_title = channel1FilesList[file_ctr][:-8].split('_')
#     file_title = file_title[0].split(' ')[4] + "_"+file_title[2] + "_" + file_title[3]
#     pearsonForEachZForEachFile.loc[file_ctr] = [file_title]+res

#
# pearsonForEachZForEachFile = pearsonForEachZForEachFile.set_index('file_name', drop=True)
# pearsonForEachZForEachFile = pearsonForEachZForEachFile.astype(float)
# f, ax = plt.subplots(figsize=(9, 6))
# sns.heatmap(pearsonForEachZForEachFile, annot=True, ax=ax)
# ax.set_title = "Pearson RedToGreen"
#
# plt.tight_layout()
# plt.show()
# # f.savefig("RedToGreen.png", dpi=200)

RADIIS = np.array(list(range(0, 2001, 200)))
MAIN_DIR_PATH = "/Users/yishaiazabary/Desktop/University/DNA Deformat proteins research/extractedStatistics/"
# GREEN_PATH = "Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)/Green Channel/Channel Image Stack/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)-Z=2.jpg"
# RED_PATH = "Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)/Red Channel/Channel Image Stack/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)-Z=2.jpg"
k_func_results_for = list()
path_to_all_channels_dirs = MAIN_DIR_PATH
channels_dirs = [dir_name[0]+"/" for dir_name in os.walk(path_to_all_channels_dirs)][1:]
channel1FilesList = [file_path[2] for file_path in os.walk(channels_dirs[0])][0]
channel1FilesList.remove('.DS_Store')
channel2FilesLisr = [file_path[2] for file_path in os.walk(channels_dirs[1])][0]
channel2FilesLisr.remove('.DS_Store')
for file_ctr in range(len(channel1FilesList)):
    first_channel_objects = pd.read_csv(channels_dirs[0] + channel1FilesList[file_ctr]).loc[:, ['X', 'Y', 'Z']]
    first_channel_objects = first_channel_objects[(first_channel_objects['Z'] >= 2) & (first_channel_objects['Z'] < 3)]
    # z_levels = first_channel_objects['Z'].astype('int').unique()
    second_channel_objects = pd.read_csv(channels_dirs[1] + channel2FilesLisr[file_ctr]).loc[:, ['X', 'Y', 'Z']]
    second_channel_objects = second_channel_objects[
        (first_channel_objects['Z'] >= 2) & (second_channel_objects['Z'] < 3)]
    if second_channel_objects.shape[0] == 0 or first_channel_objects.shape[0] == 0:
        continue
    rkf = RipleyFunction(first_channel_objects[['X', 'Y']], second_channel_objects[['X', 'Y']])
    # neighbors_for_all_spots_in_radius = rkf.calc_neighbors_in_radius_for_all_spots(radius=400)
    res = rkf.calc_neighbors_in_radii_df(RADIIS)
    k_func_results_for.append(res)
    res.to_csv("InitialResults/Ripley Results/{}.csv".format(channel1FilesList[file_ctr]))

np.save("ripleys_results_entire_dataset", np.array(k_func_results_for))
