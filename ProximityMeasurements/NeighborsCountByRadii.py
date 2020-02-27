import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv


MAIN_DIR_NAME = "/Users/yishaiazabary/Desktop/University/DNA Deformat proteins research/extractedStatistics/IsolatedNucleiStatistics/"
READ_FROM_CSV = False
# MAIN_DIR_NAME = "/Users/yishaiazabary/Desktop/University/DNA Deformat proteins research/extractedStatistics/temp/"
# READ_FROM_CSV = True
pixel_to_m_scale = 11
# an incrementing number for avoid duplication for same name experiments§§§§§§
duplication_prevention_int = 0

# radii def
RADII_STEP = 10
RADIIS = np.array(list(range(0, 700, RADII_STEP)))


def count_neighbors_in_radii(first_channel: pd.DataFrame, second_channel: pd.DataFrame):
    ans = np.zeros(shape=(len(RADIIS)))
    spots_radii_object_count = pd.DataFrame(columns=['spotID', 'radius', 'objectInRadius'])
    for index, single_spot in first_channel.iterrows():
        total_amount_of_second_spots = len(second_channel)
        distance_to_second_spots = np.sqrt(np.power(single_spot['X'] - second_channel['X'], 2) + np.power(
            single_spot['Y'] - second_channel['Y'], 2))
        for radii_index in range(len(RADIIS) - 1):
            spots_radii_object_count = pd.concat([spots_radii_object_count, pd.DataFrame(
                {"spotID": [index], 'radius': [RADIIS[radii_index]], "objectInRadius": [len(
                    distance_to_second_spots.values[(distance_to_second_spots.values >= RADIIS[radii_index]) & (
                                distance_to_second_spots.values < RADIIS[radii_index + 1])])/total_amount_of_second_spots]})])
    # averaging the number of spots in each radius
    for idx, radius in enumerate(RADIIS):
        spots_in_radii = spots_radii_object_count[spots_radii_object_count['radius'] == radius]['objectInRadius'].values
        ans[idx] = np.mean([0] if len(spots_in_radii) == 0 else spots_in_radii)
    return ans

# fig, ax = plt.subplots()
# ax.set_xlabel("Avg Distance In Pixels")
# ax.set_ylabel("Likelihood")


files_objects_in_radiis_average = dict()


if not READ_FROM_CSV:
    path_to_all_channels_dirs = MAIN_DIR_NAME
    channels_dirs = [dir_name[0] + "/" for dir_name in os.walk(path_to_all_channels_dirs)][1:]
    channel1FilesList = [file_path[2] for file_path in os.walk(channels_dirs[0])][0]
    try:
        channel1FilesList.remove('.DS_Store')
    except Exception as e:
        print("no .dstore")
    channel2FilesLisr = [file_path[2] for file_path in os.walk(channels_dirs[1])][0]
    try:
        channel2FilesLisr.remove('.DS_Store')
    except Exception as e:
        print("no .dstore")
    for file_ctr in range(len(channel1FilesList)):
        # Green to red
        try:
            first_channel_objects = pd.read_csv(channels_dirs[0] + channel1FilesList[file_ctr]).loc[:, ['X', 'Y', 'Z']]

            # if a filter to a z axis is necessary
            # first_channel_objects = first_channel_objects[(first_channel_objects['Z'] >= 2) & (first_channel_objects['Z'] < 3)]

            # z_levels = first_channel_objects['Z'].astype('int').unique()

            second_channel_objects = pd.read_csv(channels_dirs[1] + channel2FilesLisr[file_ctr]).loc[:, ['X', 'Y', 'Z']]
            # if a filter to a z axis is necessary
            # second_channel_objects = second_channel_objects[
            #     (first_channel_objects['Z'] >= 2) & (second_channel_objects['Z'] < 3)]

            # count neighbors in radius
            temp = count_neighbors_in_radii(first_channel_objects, second_channel_objects)
            # get the name of experiment from the entire file name
            # start_of_name_string = channel1FilesList[file_ctr].find('of ')+3
            # end_of_name_string = min(channel1FilesList[file_ctr].find('.nd2'), channel1FilesList[file_ctr].find('.tif'))
            file_name = channel1FilesList[file_ctr][:channel1FilesList[file_ctr].find(".nd2")-2]
            if file_name in files_objects_in_radiis_average.keys():
                file_name += "_{0}".format(duplication_prevention_int)
                duplication_prevention_int += 1
            files_objects_in_radiis_average[file_name] = temp
            print('finished file {0}'.format(file_name))
        except Exception as e:
            print("problem!:\n{0}".format(e))
    # fig_2 = pd.DataFrame(files_objects_in_radiis_average).hist(figsize=(200, 100), ax=ax, sharex=True, sharey=True)
    # normal plot for each file:

    data = pd.DataFrame(files_objects_in_radiis_average)
    data.to_csv("NeighborsCountByExperimentsDF.csv")
else:
    data = pd.read_csv("NeighborsCountByExperimentsDF.csv").drop(columns="Unnamed: 0")
    try:
        data = data.drop(columns="Unnamed: 0")
    except Exception as e:
        print("No unnamed col")

fig, axis = plt.subplots(int(len(data.keys().values)/3)+1, 3, figsize=(30, 20))
row_idx = 0 # int(len(data.keys().values)/3)+1
col_idx = 0 # int(len(data.keys().values)/3)+1
for index, file_key in enumerate(data.keys().values):
    axis[col_idx][row_idx].plot(
        ['{0}-{1}µM'.format(x*RADII_STEP*pixel_to_m_scale, (x+1)*RADII_STEP*pixel_to_m_scale) for x in np.arange(0, len(data[file_key].values), 1)],
        data[file_key].values)
    axis[col_idx][row_idx].tick_params(axis='x', labelsize=4, rotation=30)
    axis[col_idx][row_idx].set_ylim(0, 0.2)
    axis[col_idx][row_idx].set_title("{0}".format(file_key), fontsize=10)
    row_idx += 1
    if (row_idx % 3) == 0:
        row_idx = 0
        col_idx += 1
# pd.DataFrame(files_objects_in_radiis_average).to_csv("test.csv")
# [x.title.set_size(3) for x in fig_2.ravel()]

plt.tight_layout()
plt.show()
text = input("Save? Y/N   :")
if text.lower() == 'y':
    fig.savefig('InitialResults/ProteinsProximity/NucleiOnlyRedLikelihoodToAppearNextToGreen.png', dpi=300)



# todo: each pixel is approx 11 µM

# first option - מעכב לאחד החלבונים ולבדוק יחסים בינו לבין השני
# second option - שינוי גנטי, עלול לעורר בעיות וייקח חודשים עד חצי שנה להרים את הדאטה
# use only the spots inside the nuclei

