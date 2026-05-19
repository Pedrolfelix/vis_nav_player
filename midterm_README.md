# 🧭 Midterm — Visual Maze Navigation System

> **ROB-GY 6203: Robot Vision** | NYU Tandon School of Engineering

A vision-only navigation system that guides a robot through an unknown maze using only first-person camera images — no odometry, no IMU, no GPS.

---

## 🎯 Problem Statement

Given a sequence of exploratory images of a maze environment, build a system that:

- **Builds a visual map** of the explored space without any pose sensors
- **Localizes the robot in real-time** from a single first-person frame
- **Plans and executes a path** from the current position to a goal defined only by target images
- **Recovers from drift, occlusion, and dead-ends** without ever knowing absolute pose

---

## 🧠 Approach

### 1. Visual Feature Extraction — DINOv2
Used Meta AI's **DINOv2** as the backbone for visual feature extraction. DINOv2's self-supervised pretraining gives it strong performance across diverse textures and lighting conditions, making it ideal for the varied visual appearance of maze environments.

### 2. Efficient Similarity Search — FAISS
All extracted image embeddings are indexed using **FAISS** (Facebook AI Similarity Search), enabling fast nearest-neighbor lookup to identify the closest known location to the current frame.

### 3. Visual Map Construction
During the exploration phase, the system incrementally builds a map by associating each captured frame with its FAISS-indexed embedding — creating a purely visual topological map.

### 4. Real-Time Localization
At navigation time, a single query frame is embedded and searched against the map index to retrieve the most visually similar known node — establishing the robot's current location without any external sensors.

### 5. Path Planning & Visualization
**Pygame** was used for real-time visualization of the robot's state and for the manual interaction interface during development and testing.

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| **DINOv2** (Meta AI) | Visual feature extraction backbone |
| **FAISS** | Billion-scale similarity search for image retrieval |
| **OpenCV** | Image preprocessing and computer vision utilities |
| **NumPy** | High-performance numerical operations |
| **Pygame** | Real-time visualization and interaction interface |

---

## 📚 References

- Arandjelović et al., *NetVLAD: CNN for weakly supervised place recognition*, CVPR 2016
- Jégou et al., *Aggregating Local Descriptors into a Compact Image Representation*, CVPR 2010
- Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs*, IEEE TBD 2019
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Pedrolfelix/vis_nav_player.git
cd vis_nav_player/midterm

# Install dependencies
pip install -r requirements.txt

# Run the navigation player
python player.py
```

> ⚠️ **Note:** DINOv2 model weights are downloaded automatically on first run via `torch.hub`. Requires internet connection and ~1GB of disk space.

---

## 📊 Results

<!-- Add your demo video, GIF, or screenshots here -->
> 📹 Demo video: *[link to video]*  
> 📊 Presentation: *[link to slides]*

---

## 👤 Author

**Pedro Felix** | NYU Tandon School of Engineering  
[GitHub](https://github.com/Pedrolfelix)
