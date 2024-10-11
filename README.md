# Uncertainty-Aware Motion Planning for Ground Vehicles

This repository contains the implementation of a framework for **Uncertainty-Aware Motion Planning in Unstructured and Uneven Off-Road Terrain**. The framework assesses terrain using geometric and semantic data, clusters uncertain regions, and generates a stochastic graph. A multi-query planner explores potential routes, and the Complete CAO* (CCAO*) algorithm is used to ensure optimal navigation, even when deterministic paths don't exist.


## Features
- Geometric & Semantic Terrain Data Integration: Combines slope, roughness, and terrain semantics for accurate traversability assessment.
- Unsupervised Region Clustering: Groups grid cells with similar spatial and visual features into uncertain regions.
- Stochastic Graph Representation: Converts uncertain environments into stochastic graphs for navigation.
- Complete CAO (CCAO) Algorithm: Guarantees pathfinding solutions for stochastic graphs even in cases where no deterministic path exists.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/uncertainty-aware-motion-planning.git
   cd uncertainty-aware-motion-planning
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   sudo apt install graphviz

## Usage

### Step 1: Get Images and Digital Elevation Maps (DEM) from Point Cloud

1. Open the point cloud file in **CloudCompare**.
2. Go to the **Tools** tab, select **Projection**, and choose **Rasterize (and contour plot)**.
3. Choose the step size as per your requirement.
4. In the **Projection** section, set the direction to **Z** and configure the cell height. Set the **Project SF(s)** to **Average**.
5. Click **Update Grid** to apply the settings.
6. To export the grid, click **Raster** at the bottom and save it as a `.tif` file.

- For a top view image of the environment with the same resolution:
    - Change the active layer to **RGB**.
    - Click on **Image** at the bottom to export the grid as an image.

Now, you have both a **Digital Elevation Map (DEM)** and a top view image of the environment.

### Step 2: Get Terrain's Semantic Features Using Roboflow

1. **Create a Roboflow Account**:
   - Go to [Roboflow](https://roboflow.com/), sign up, and create a new project.

2. **Upload Images**:
   - Upload the top-view images of the terrain (from Step 1) to your Roboflow project.

3. **Annotate for Semantic Segmentation**:
   - Select **Semantic Segmentation** as your annotation type.
   - Define labels for terrain features (e.g., grass, water, rocks).
   - Manually annotate each feature by drawing polygons over the corresponding regions or by using the smart polygon tool on the images.

4. **Generate Dataset and Export Semantic Segmentation Masks**:
   - Once the annotation is complete, generate a dataset from the annotated images.
   - Roboflow will automatically generate the **semantic segmentation masks** based on your annotations.
   - Download the dataset and choose to download the Semantic Segmentation Masks in a zip file on your computer

     
Now, you have **semantic segmentation masks** for the terrain features, which can be used in subsequent processing steps.

### Step 3: Get the Geomatric Traversability Map

1. **Prepare the Data**:
   - Place the **DEM** and **top-view image** files (from Step 1) and the **semantic segmentation mask** (from Step 2) into the `data/` folder of your project.

2. **Update Paths in the Configuration**:
   - Navigate to the `config/` folder and open the `config.yaml` file.
   - Update the file paths to point to the correct DEM, top-view image, and semantic segmentation mask files.

3. **Run the Geometric Analysis**:
   - Execute the `dem_analysis.py` script to process the DEM and generate the **geometric classification** of the environment.
   - The script will analyze the terrain data and save a `.pickle` file in the `results/` folder, containing the geometric classification results.

Now, you have the **traversability map** generated based on both the terrain's geometric and semantic features.

### Step 4: Generate Uncertain Regions

1. **Prepare the Image for Segmentation**:
   - Copy the **image** you want to segment (from Step 1 or Step 2) into the `data/` folder inside the `superpixel_annotation/` directory.

2. **Run the Superpixel Annotation Script**:
   - Execute the `superpixel_annotation.py` script located in the `superpixel_annotation/` folder:
     ```bash
     python superpixel_annotation/superpixel_annotation.py
     ```

3. **Select the Segmentation Method**:
   - After running the script, open the `data/` folder inside the `superpixel_annotation/` directory.
   - Choose a **segmentation method** from the available options.

4. **Select Regions and Get Coordinates**:
   - Click on the regions you wish to segment in the image to obtain their coordinates.

5. **Save the Coordinates**:
   - Copy the selected region's coordinates and paste them into the `regions.txt` file located in the `data/` folder.

Now, you have generated the **uncertain regions** by segmenting the image and saving their coordinates for use in further analysis.



