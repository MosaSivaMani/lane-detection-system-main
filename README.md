
# 🚗 Lane Detection System

This project implements a **lane detection system** using Python and OpenCV. It processes video footage (or optionally webcam feed) to detect road lane lines in real-time and overlays them onto the video. This type of application is commonly used in driver assistance systems.

---

## 📁 Project Structure

```

lane-detection-system-main/
│
├── lanedet.py                # Main Python script for lane detection
├── sampleVideo.mp4           # Sample input video file (add your own)
├── Output\_Lane\_det.mp4       # Output video file with detected lanes
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation

````

---

## 🧰 Technologies Used

- **Python 3.6+**
- **OpenCV** – for image and video processing
- **NumPy** – for numerical operations

---

## 🚀 Features

- Canny Edge Detection for identifying edges in frames  
- Region of Interest (ROI) masking to focus on road area  
- Hough Transform to detect straight lane lines  
- Line averaging and overlay onto original frames  
- Real-time video processing and MP4 output  

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lane-detection-system-main.git
cd lane-detection-system-main
````

### 2. Create and Activate Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python numpy
```

---

## 🎥 How to Run

### 📼 Using Sample Video

Ensure `sampleVideo.mp4` is in the same folder. Then run:

```bash
python lanedet.py
```

* Output will be saved to: `Output_Lane_det.mp4`
* Detected lane lines will be shown live in a window

### 🎥 Optional: Use Webcam Instead of Video

In `lanedet.py`, change the video capture line:

```python
cap = cv2.VideoCapture(0)  # Use webcam
```

Then rerun:

```bash
python lanedet.py
```

---

## 📤 Output


https://github.com/user-attachments/assets/877edb2e-d25e-4b71-ba18-73de217a242f

* Detected lanes will be drawn in **green**
* Final processed video will be saved as `Output_Lane_det.mp4`
* Press `q` to quit during live display

---

## 🧠 Algorithm Breakdown

1. **Grayscale Conversion**
2. **Gaussian Blur**
3. **Canny Edge Detection**
4. **Region of Interest Masking**
5. **Hough Line Detection**
6. **Line Averaging and Drawing**

---

## 📸 Example Frame (Visualization)

```
[Frame from video]
+ Green lines overlaid along left/right lane boundaries
```

---

## 💡 Future Enhancements

* Handle curved lanes using polynomial fitting
* Detect lane departure and alert driver
* Integrate vehicle steering angle prediction
* Upgrade to deep learning (e.g., U-Net for segmentation)

---

## 📝 License

This project is licensed under the MIT License.

---

## 👨‍💻 Contact
GitHub: [@MosaSivaMani](https://github.com/MosaSivaMani)
Email: [mosasiva6@gmail.com](mosasiva6@gmail.com)

```

---
