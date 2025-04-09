# TypeVision

**TypeVision** is a computer vision-based system that enables virtual interaction with a keyboard and mouse using only a webcam. By leveraging the power of MediaPipe and OpenCV, it tracks hand movements in real-time and translates them into keyboard presses or mouse actions on a printed keyboard image. This makes it possible to interact with a computer without physical input devices.
The system uses visual cues to identify the location of a printed keyboard, detect finger movements, and analyze hand gestures. It supports various features including typing, mouse movement tracking, and virtual mouse clicking.

## Usage

```bash
python -m Project_A [options]
```

### Options:

| Flag                | Description                                 |
|---------------------|---------------------------------------------|
| `-k` / `--activate_keyboard`       | Enable keyboard typing detection          |
| `-mm` / `--activate_mouse_movement` | Enable mouse movement tracking            |
| `-mc` / `--activate_mouse_click`    | Enable mouse click detection              |
| `-hk` / `--hide_keyboard_image`     | Hide the keyboard visualization           |
| `-hm` / `--hide_mouse_image`        | Hide the mouse status overlay             |
| `-hc` / `--hide_camera_image`       | Hide the camera feed                      |
| `--id`                              | Camera ID (default: 0)                    |
| `--video-path`                      | Use video file instead of live camera     |

### Example:

```bash
# activate keyboard recognition from default web cam
python -m Project_A --activate_keyboard --id 0

# activate mouse tracking and clicking recognition from external web cam
python -m Project_A --activate_mouse_movement --activate_mouse_click --id 1

# activate keyboard recognition from pre-recorded images folder
python -m Project_A -activate_keyboard --video-path captured_images
```

---

## 📦 Environment Setup

This project uses **Conda** to manage dependencies.

### 1. Create and activate the Conda environment:

```bash
git clone https://github.com/orslup/Project_A.git
cd Project_A
conda create -n keyboard_mouse python=3.10
conda activate keyboard_mouse
```

### 2. Install the dependencies:

```bash
pip install -r requirements.txt
```

