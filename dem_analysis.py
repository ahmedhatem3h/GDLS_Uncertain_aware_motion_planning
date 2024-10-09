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
img_path = os.path.join(current_directory, "..", 'maps', 'HO_P1_DEM.png')
image = cv2.imread(img_path)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


dem_path = os.path.join(current_directory, "..", 'maps', 'HO_P1_DEM.tif')
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


# mask = np.logical_and( slope > 25, slope < 65)
# img_slope_thresh = img * mask.data[:, :, None]
# figure = plt.figure()
# mask.plot()



# figure = plt.figure()
# segments = quickshift(slope.data, kernel_size=5, max_dist=8, ratio=0.5, convert2lab=False)
# print(f'Quickshift number of segments: {len(np.unique(segments))}')
# # segments = slic(mask.data, n_segments=100, compactness=0.5, sigma=1, start_label=1, channel_axis=None)
# plt.imshow(mark_boundaries(slope.data/np.max(slope.data), segments), cmap="Reds")


ruggedness = xdem.terrain.terrain_ruggedness_index(dem)
print(np.max(ruggedness), np.min(ruggedness))
plot_attribute(ruggedness, "Reds", vlim=(0, 4.4))


# roughness = xdem.terrain.roughness(dem)
# print(np.max(roughness), np.min(roughness))
# plot_attribute(roughness, "Reds", "Roughness", vlim=(0, 2.4))

# mask = np.logical_and( roughness > 0.3, roughness < 0.7)
# img_slope_thresh = img * mask.data[:, :, None]
# figure = plt.figure()
# mask.plot()




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
# set cell 00 to 0
mask_final[0, 0] = 4


traversability = 0.5 * (1.0 - (slope / 40)) + 0.5 * (1.0 - (ruggedness / 1.4))
traversability = traversability.data
traversability = np.clip(traversability, 0, 1)

#  if slope is greater than 40, then set it to 0, if  ruggedness is greater than 1.0, then set it to 0
traversability = np.where(slope.data > 40, 0, traversability)    
traversability = np.where(ruggedness.data > 1.4, 0, traversability)
nan_mask = np.logical_or(np.isnan(slope.data), np.isnan(ruggedness.data))
traversability = np.where(nan_mask, np.NaN, traversability)
rev_traversability = 1 - traversability

# if value is not defined in slope or ruggedness, then make it not defined in traversability


# traversability = np.where(slope.data < 15, 1, traversability)

current_directory = os.getcwd()
image_path = os.path.join(current_directory, "..", 'maps', 'HO_P1_DEM_mask_2.png')
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


final_traversability_classes = mask_final.copy()

# if value of pixel a value from the untraversable classes, then it is an obstacle
mask_obstacle = np.isin(img, semantic_untraversable_class_list)
final_traversability_classes[mask_obstacle.data] = 0
nan_mask = np.logical_or(np.isnan(slope.data), np.isnan(ruggedness.data))
# final_traversability_classes = np.where(nan_mask, np.NaN, final_traversability_classes)
# final_traversability_classes[0, 0] = 4



figure = plt.figure()
plt.axis('off')
plt.tight_layout()
plt.imshow(final_traversability_classes, cmap='gray')
# plt.imshow(mask_final, cmap='gray')
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

# scale = 1
# final_traversability_classes, cost_map, contours_objects, obstacle_list = get_map(scale = scale)

# for j in range(len(contours_objects)):
#     xx, yy  = contours_objects[j].polygon.exterior.xy
#     xx = [x*scale for x in xx]
#     yy = [y*scale for y in yy]
#     plt.fill(xx, yy , 'r', alpha=0.7)

# figure = plt.figure()
# plt.axis('off')
# plt.imshow(rev_traversability, cmap="Reds")
# # plt.show()





# ruggedness = ruggedness.data
# ruggedness = np.clip(ruggedness, 0, 1.4)
# plt.imshow(ruggedness, cmap="Reds",  add_cbar=True, cbar_title="Ruggedness")

# # traversability = 0.5 * (1.0 - (slope / 40)) + 0.5 * (1.0 - (ruggedness / 1.4))
# # traversability = traversability.data
# # traversability = np.clip(traversability, 0, 1)
# plt.imshow(traversability, cmap="Reds",  add_cbar=True, cbar_title="Traversability")




#  if values is greater than 1, then set it to 1, if less than 0, then set it to 0, dont use np.clip or np.where




# plot_attribute(traversability, "gray", "Traversability")


pickle.dump(mask_final, open(r'results/mask_final.pickle', 'wb'))


# uncertain_regions = mask_final == 1
# uncertain_regions = uncertain_regions.astype(np.uint8)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# uncertain_regions = cv2.morphologyEx(uncertain_regions, cv2.MORPH_OPEN, kernel)



# contours, _ = cv2.findContours(uncertain_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# new_contours = []
# # loop on contours and if area is less than 100, then remove it
# for contour in contours:
#     if cv2.contourArea(contour) > 50:
#         new_contours.append(contour)
#         cv2.drawContours(mask_final, [contour], -1, (5, 0, 0), 1)

# print(len(new_contours))
# # draw the contours on the image
# # cv2.drawContours(mask_final, contours, -1, (5, 0, 0), 1)


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
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()








# # mask_safe = np.logical_and( slope < 25, roughness < 0.25)
# # mask_obstacle = np.logical_or( slope > 50, roughness > 0.9)

# # risky is the remaining area
# mask_safe = np.logical_and( slope < 25, roughness < 0.25)
# mask_obstacle = np.logical_or( slope > 50, roughness > 0.9)
# mask_risky = np.logical_and( np.logical_not(mask_safe), np.logical_not(mask_obstacle))

# # plot the three masks on the same figure with different colors

# figure = plt.figure()
# plt.imshow(mask_safe.data, cmap="Greens")
# plt.imshow(mask_obstacle.data, cmap="Reds", alpha=0.5)
# plt.imshow(mask_risky.data, cmap="Blues", alpha=0.5)
# plt.colorbar(label="Mask")
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()



# # # check if the masks are disjoint
# # assert np.all(np.logical_and(mask_safe.data, mask_obstacle.data) == 0)
# # assert np.all(np.logical_and(mask_safe.data, mask_risky.data) == 0)
# # assert np.all(np.logical_and(mask_obstacle.data, mask_risky.data) == 0)
# # # check if the masks are exhaustive
# # assert np.all(np.logical_or(np.logical_or(mask_safe.data, mask_obstacle.data), mask_risky.data))


# # get the contours for the obstacle mask and fill the contours
# contours, _ = cv2.findContours(mask_obstacle.data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# mask_obstacle_contours = np.zeros_like(mask_obstacle.data).astype(np.uint8)
# cv2.drawContours(mask_obstacle_contours, contours, -1, 1, thickness=cv2.FILLED)
# # plot the obstacle mask and the contours
# figure = plt.figure()
# plt.imshow(mask_obstacle_contours, cmap="Greens")
# plt.colorbar(label="Obstacle Mask")
# plt.xticks([])
# plt.yticks([])


# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
# mask_risky_contours = cv2.morphologyEx(mask_risky.data.astype(np.uint8), cv2.MORPH_OPEN, kernel)

# # # mask_risky = np.logical_and(mask_risky.data, np.logical_not(mask_obstacle_contours))
# # # get the contours for the risky mask and fill the contours
# # contours, _ = cv2.findContours(mask_risky.data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # mask_risky_contours = np.zeros_like(mask_risky.data).astype(np.uint8)
# # cv2.drawContours(mask_risky_contours, contours, -1, 1, thickness=cv2.FILLED)
# # mask_risky_contours = np.logical_and(mask_risky_contours, np.logical_not(mask_obstacle_contours))

# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
# # mask_risky_contours = cv2.morphologyEx(mask_risky_contours.astype(np.uint8), cv2.MORPH_OPEN, kernel)

# contours, _ = cv2.findContours(mask_risky_contours.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)


# # figure = plt.figure()
# # segments = quickshift(mask_risky_contours, kernel_size=5, max_dist=8, ratio=0.5, convert2lab=False)
# # print(f'Quickshift number of segments: {len(np.unique(segments))}')
# # # segments = slic(mask.data, n_segments=100, compactness=0.5, sigma=1, start_label=1, channel_axis=None)
# # plt.imshow(mark_boundaries(mask_risky_contours, segments), cmap="Reds")

# figure = plt.figure()
# plt.imshow(mask_risky_contours, cmap="Greens")
# plt.colorbar(label="Risky Mask")
# plt.xticks([])
# plt.yticks([])

# # plot the obstacle mask and the contours
# figure = plt.figure()
# plt.imshow(img)
# # plt.imshow(mask_risky_contours2)


# figure = plt.figure()
# plt.imshow(mask_obstacle_contours, cmap="Reds", alpha=0.5)
# plt.imshow(mask_risky_contours, cmap="Blues", alpha=0.5)


# # create the final mask where if true in obstacle mask, then 1, if true in risky mask, then 0.5, else 0
# mask_final = np.ones_like(mask_risky_contours) * 3
# mask_final[mask_safe.data] = 2
# mask_final[mask_risky.data] = 1
# mask_final[mask_obstacle.data] = 0

# # mask_final[mask_risky_contours == 1] = 1
# # mask_final[mask_obstacle_contours == 1] = 0
# # print(mask_risky_contours == 1)


# figure = plt.figure()
# plt.imshow(mask_final, cmap='gray')
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()


# # generate the final mask where safe is 1, risky is 2 and obstacle is 3
# mask_final = np.zeros_like(mask_safe.data)
# mask_final[mask_safe.data] = 0
# mask_final[mask_obstacle.data] = 1
# mask_final[mask_risky.data] = 0.5

# figure = plt.figure()
# plt.imshow(mask_final, cmap="Grays", vmin=0, vmax=1)
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()


# # generate the final mask where safe is 1, risky is 2 and obstacle is 3
# mask_final = np.zeros_like(mask_safe.data)
# mask_final[mask_safe.data] = 0
# mask_final[mask_obstacle.data] = 1

# figure = plt.figure()
# plt.imshow(mask_final, cmap="coolwarm", vmin=0, vmax=1)
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()


# roughness_thresh = roughness < 0.3
# roughness_thresh.plot()


# figure = plt.figure()
# segments = quickshift(roughness.data, kernel_size=5, max_dist=0.1, ratio=0.5, convert2lab=False)
# print(f'Quickshift number of segments: {len(np.unique(segments))}')
# # segments = slic(mask.data, n_segments=100, compactness=0.5, sigma=1, start_label=1, channel_axis=None)
# plt.imshow(mark_boundaries(roughness.data/np.max(roughness.data), segments), cmap="Reds")





# print(np.max(traversability), np.min(traversability). np.mean(traversability))


# mask1 = traversability > 0.2
# mask2 = traversability > 0.3
# mask3 = traversability > 0.6
# figure = plt.figure()
# mask1.plot()
# figure = plt.figure()
# mask2.plot()
# figure = plt.figure()
# mask3.plot()

# mask4 = np.logical_and( traversability > 0.2, traversability < 0.6)
# figure = plt.figure()
# mask4.plot()


# mask4_data = mask4.data
# # Perform morphology operations to remove noise and detect contours
# opened_mask = binary_opening(mask4_data, structure=np.ones((3, 3)))
# closed_mask = binary_closing(opened_mask, structure=np.ones((3, 3)))

# # Plot the final mask and contours
# figure = plt.figure()
# plt.imshow(closed_mask, cmap="gray")
# plt.colorbar(label="Mask")
# # plt.contour(contours, colors='red', linewidths=1)
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()

plt.show()