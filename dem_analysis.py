import xdem
import os
import rasterio
import cv2
import pickle
import math

import matplotlib.pyplot as plt
import numpy as np

from utils import param_loader

# Load parameters
params = param_loader()
semantic_untraversable_class_list = params["semantic_untraversable_class_list"]
dem_file_path = params["dem_file_path"]
top_view_image_path = params["top_view_image_path"]
semantic_segmentation_mask_path = params["semantic_segmentation_mask_path"]

def plot_attribute(attribute, cmap, label=None, vlim=None):

    add_cbar = True if label is not None else False

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    plt.axis('off')
    if vlim is not None:
        if isinstance(vlim, (int, float)):
            vlims = {"vmin": -vlim, "vmax": vlim}
        elif len(vlim) == 2:
            vlims = {"vmin": vlim[0], "vmax": vlim[1]}
    else:
        vlims = {}

    attribute.plot(ax=ax, cmap=cmap, add_cbar=add_cbar, cbar_title=label, **vlims)

    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()


current_directory = os.getcwd()
img_path = os.path.join(current_directory, top_view_image_path)
image = cv2.imread(img_path)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


dem_path = os.path.join(current_directory, dem_file_path)
src = rasterio.open(dem_path)
array = src.read(1)
dem = xdem.DEM.from_array(array, transform=src.transform, crs="EPSG:4326", nodata=src.nodata)
plot_attribute(dem, "jet")
print(np.max(dem.data), np.min(dem.data))
# plot_attribute(dem, "rainbow", "Elevation (m)")

slope = xdem.terrain.slope(dem)
# slope_array = slope.data
print(np.max(slope), np.min(slope))
plot_attribute(slope, "Reds", vlim=(0, 50))


ruggedness = xdem.terrain.terrain_ruggedness_index(dem)
print(np.max(ruggedness), np.min(ruggedness))
plot_attribute(ruggedness, "Reds", vlim=(0, 4.4))



class1_mask = np.logical_and( slope < 15, ruggedness < 0.15)
# class 2 mask if slope is bet 15 and 35 or ruggedness is bet 0.15 and 0.4
class2_mask = np.logical_or( np.logical_and(slope > 15, slope < 30), np.logical_and(ruggedness > 0.15, ruggedness < 0.6))
# class 3 mask if slope is bet 35 and 40 or ruggedness is bet 0.4 and 0.7
class3_mask = np.logical_or( np.logical_and(slope > 30, slope < 40), np.logical_and(ruggedness > 0.6, ruggedness < 1.4))
# class 4 if slope is greater than 40 or ruggedness is greater than 0.9
class4_mask = np.logical_or( slope > 40, ruggedness > 1.4)

# plot the three masks on the same figure with different colors
mask_final = np.ones_like(class1_mask.data.astype(np.uint8)) * 4
mask_final[class1_mask.data] = 3
mask_final[class2_mask.data] = 2
mask_final[class3_mask.data] = 1
mask_final[class4_mask.data] = 0


pickle.dump(mask_final, open(r'results/geometric_classification.pickle', 'wb'))


figure = plt.figure()
plt.imshow(mask_final, cmap='gray')
plt.axis('off')
plt.tight_layout()
# Add legend where pixels with color of 0 is called s4, 1 is called s3, 2 is called s2, 3 is called s1
import matplotlib.patches as mpatches
legend_labels = {
    3: "s$_1$",
    2: "s$_2$",
    1: "s$_3$",
    0: "s$_{untrav}$"
}

# Create custom legend with square patches
handles = [mpatches.Patch(color=plt.cm.gray(i / 4.0), label=legend_labels[i], edgecolor='black', linewidth=1) for i in range(4)]
plt.legend(handles=handles, loc='upper right')


# traversability = 0.5 * (1.0 - (slope / 40)) + 0.5 * (1.0 - (ruggedness / 1.4))
# traversability = traversability.data
# traversability = np.clip(traversability, 0, 1)

# #  if slope is greater than 40, then set it to 0, if  ruggedness is greater than 1.0, then set it to 0
# traversability = np.where(slope.data > 40, 0, traversability)    
# traversability = np.where(ruggedness.data > 1.4, 0, traversability)
# nan_mask = np.logical_or(np.isnan(slope.data), np.isnan(ruggedness.data))
# traversability = np.where(nan_mask, np.NaN, traversability)
# rev_traversability = 1 - traversability


current_directory = os.getcwd()
image_path = os.path.join(current_directory, semantic_segmentation_mask_path)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


final_traversability_classes = mask_final.copy()

# if value of pixel a value from the untraversable classes, then it is an obstacle
mask_obstacle = np.isin(img, semantic_untraversable_class_list)
final_traversability_classes[mask_obstacle.data] = 0
nan_mask = np.logical_or(np.isnan(slope.data), np.isnan(ruggedness.data))



figure = plt.figure()
plt.axis('off')
plt.tight_layout()
plt.imshow(final_traversability_classes, cmap='gray')
# Add legend where pixels with color of 0 is called s4, 1 is called s3, 2 is called s2, 3 is called s1
import matplotlib.patches as mpatches
legend_labels = {
    3: "s$_1$",
    2: "s$_2$",
    1: "s$_3$",
    0: "s$_{untrav}$"
}
# Create custom legend with specific colors
colors = ['black', 'gray', 'lightgray','white']
handles = [mpatches.Patch(color=colors[i], label=legend_labels[i], edgecolor='black', linewidth=1) for i in range(4)]
plt.legend(handles=handles, loc='upper right')












plt.show()