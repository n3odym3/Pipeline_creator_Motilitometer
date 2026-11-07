# Pipeline Creator - Motilitometer

Control and data processing interface for the **Motilitometer**, an open-source, modular digital microscopy platform designed for motility and photoresponse analysis of swimming microorganisms (e.g., microalgae).

This software is built using [Pipeline_creator](https://github.com/n3odym3/Pipeline_creator).

---

## 🔬 Scientific Context

This project provides the control interface, hardware orchestration, and video analysis pipeline described in:

> **Motilitometer: A compact modular microscope for motility and photoresponse analysis of microorganisms**  
> *HardwareX*, Volume 26, 2026, e00788.  
> 📄 **Article:** [ScienceDirect (PII: S2468067226000489)](https://www.sciencedirect.com/science/article/pii/S2468067226000489)

---

## 🚀 Key Features

- **Hardware Control:** Camera stream acquisition and synchronized, programmable multispectral / directional LED stimulations.
- **Motion & Behavioral Analysis:**
  - Real-time and offline video processing.
  - Adaptive contour detection, thresholding, and surface coverage.
  - Multi-target tracking and velocity vector computation.
  - Dense optical flow (Farneback) and sedimentation/drift compensation.
  - Angular distributions, Polar plot, spectrograph, and Phototaxis Index calculation.

---

## 💻 Usage

Launch the main interface:

```bash
python main.py
```

The preconfigured Motilitometer pipeline (`pipelines/Motilitometer.json`) automatically wires the camera acquisition, tracking processors, and visualization widgets.

---

