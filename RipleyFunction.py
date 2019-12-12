import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv


class RipleyFunction:

    def __init__(self, spots_typeA: pd.DataFrame, spots_typeB: pd.DataFrame, plot_x_len: int, plot_y_len: int):
        self.spots_typeA = spots_typeA
        self.spots_typeB = spots_typeB
        self.plot_x_len = plot_x_len
        self.plot_y_len = plot_y_len

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
            # todo: ask assaf of the inverse cosine x being more than 1.
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
                "k_for_radius": math.pow(self.spots_typeB.shape[0], -2) *
                                2 * math.pi * radius *
                                np.sum(np.array([x.get('weighted_distance') for x in number_of_typeB_spots_in_radius_for_all_spots]))

            })
        return results_for_radii

    def calc_neighbors_in_radii_df(self, radii):
        res_as_dictionaries = self.calc_neighbors_in_radii(radii=radii)
        radius = np.array([x.get('radius') for x in res_as_dictionaries])
        k = np.array([x.get('k_for_radius') for x in res_as_dictionaries])
        return pd.DataFrame({'Radius': radius, "K": k})






GREEN_PATH = "Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)/Green Channel/Channel Image Stack/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)-Z=2.jpg"
RED_PATH = "Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)/Red Channel/Channel Image Stack/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)-Z=2.jpg"

# Z=3
# GREEN_PATH = "Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)/Green Channel/Channel Image Stack/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)-Z=3.jpg"
# RED_PATH = "Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)/Red Channel/Channel Image Stack/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)-Z=3.jpg"


green_image = cv.imread(GREEN_PATH)
red_image = cv.imread(RED_PATH)
green_channel = pd.read_csv("Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)/Green Channel/Statistics for Median of A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)-1.csv")
red_channel = pd.read_csv("Extracted spots detection by FIJI/A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)/Red Channel/Statistics for Median of A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 - A5_M_H3K27ac_488_Rab_H3K4me1_546_001.nd2 (series 1)-1.csv")


rkf = RipleyFunction(green_channel[['X', 'Y']], red_channel[['X', 'Y']], plot_x_len=green_image.shape[0], plot_y_len=green_image.shape[1])
# neighbors_for_all_spots_in_radius = rkf.calc_neighbors_in_radius_for_all_spots(radius=400)
print(rkf.calc_neighbors_in_radii_df([300, 320]).head())


