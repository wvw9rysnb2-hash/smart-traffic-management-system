# Smart Traffic Management System 🚦

## Project Overview
The Smart Traffic Management System is a Python-based application that uses Artificial Intelligence and Computer Vision to detect and count vehicles from uploaded traffic videos.

The system uses the YOLOv8 object detection model to identify different vehicle categories and ByteTrack to track vehicles and count unique detections.

## Features
- Upload traffic videos in MP4, AVI, MOV and MKV formats.
- Detect cars, buses, trucks, bicycles and motorcycles.
- Track vehicles using unique tracking IDs.
- Display vehicle counts.
- Generate bar charts and pie charts for traffic analysis.
- Interactive dashboard built with Streamlit.

## Technologies Used
- Python
- Streamlit
- OpenCV
- YOLOv8
- ByteTrack
- Pandas
- Plotly

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run stmsa.py
```

## Project Objective
To demonstrate how Artificial Intelligence and Computer Vision can support traffic monitoring through automated vehicle detection, tracking and analysis.

## Author
Diploma in Computer Science and Engineering Student

## Project Type
Academic Project – Smart Traffic Management System
