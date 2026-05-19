# 🏁 Final Project — Autonomous Visual Maze Solver

> **ROB-GY 6203: Robot Vision** | NYU Tandon School of Engineering

Full autonomous maze navigation system built on top of classical and learned visual place recognition techniques — the robot builds its own map, localizes itself, and finds a path to the goal using only a camera.

---

## 🎯 Problem Statement

Given only a set of exploratory images captured during a first run through the maze, design a complete navigation system that:

- **Builds a visual map** of the explored space without odometry, IMU, or GPS
- **Localizes the robot in real-time** from a single first-person frame
- **Plans and executes a path** from the start node to a goal defined only by target images
- **Recovers from drift, occlusion and dead-ends** without ever knowing absolute pose

---

## 🧠 Approach

### Feature Extraction & Description

**ORB (Oriented FAST and Rotated BRIEF)**  
Used as the primary local feature detector and descriptor. ORB provides fast, rotation-invariant keypoint detection and binary descriptors suitable for real-time matching.

**SIFT (Scale-Invariant Feature Transform)**  
Applied for more robust feature matching in ambiguous visual environments where scale changes pose a challenge.

### Image Representation

**VLAD (Vector of Locally Aggregated Descriptors)**  
Local descriptors (ORB/SIFT) are aggregated into a compact global image representation using VLAD encoding. A visual vocabulary is built from the exploration frames via k-means clustering, and each image is encoded as the sum of residuals between its descriptors and the assigned cluster centers.

### Similarity Search

**FAISS**  
VLAD vectors are indexed using FAISS for efficient approximate nearest-neighbor search, enabling fast localization even with large map sizes.

### Navigation Pipeline

```
Exploration Phase:
  Capture frames → Extract ORB/SIFT features → Encode with VLAD → Index with FAISS → Build topological map

Navigation Phase:
  Capture query frame → Extract features → Encode → FAISS lookup → Retrieve closest map node → Plan path → Execute
```

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| **ORB** (OpenCV) | Fast local feature detection and description |
| **SIFT** (OpenCV) | Scale-invariant feature matching |
| **VLAD** | Compact global image representation via descriptor aggregation |
| **FAISS** | Efficient similarity search over VLAD vectors |
| **OpenCV** | Core computer vision pipeline |
| **NumPy / scikit-learn** | K-means for visual vocabulary, numerical ops |
| **Pygame** | Visualization and interaction |

---

## 📚 References

- Jégou et al., *Aggregating Local Descriptors into a Compact Image Representation*, CVPR 2010
- Arandjelović et al., *NetVLAD: CNN for weakly supervised place recognition*, CVPR 2016
- Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004
- Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs*, IEEE TBD 2019
- Hartley & Zisserman, *Multiple View Geometry* — essential matrix & pose recovery
- Grisetti et al., *A Tutorial on Graph-based SLAM*, IEEE ITS 2010

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Pedrolfelix/vis_nav_player.git
cd vis_nav_player/final_project

# Install dependencies
pip install -r requirements.txt

# Run exploration phase to build the map
python explore.py

# Run the autonomous navigator
python player.py
```

---

## 📊 Results

<!-- Add your demo video, GIF, or screenshots here -->
> 📹 Demo video: *[link to video]*  
> 📊 Final presentation: *[link to slides]*  
> 🖼️ Screenshots: *[add images below]*

---

## 👤 Author

**Pedro Felix** | NYU Tandon School of Engineering  
[GitHub](https://github.com/Pedrolfelix)
