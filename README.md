
# Real-Time Facial Recognition Project

Author: Vitor Mendes  
Email: vitor.mendes@ieee.org

## Description

This repository contains a real-time facial recognition project using a deep learning model. The project includes scripts for real-time video capture, face classification, and a pre-trained model for facial recognition.

## Repository Structure

- **main.py**: The main script that initializes the facial recognition system.
- **face_recognition.py**: Script responsible for capturing real-time video and performing facial recognition.
- **face_recognition_model.h5**: Pre-trained model file for facial recognition.
- **class_labels.npy**: File containing the class labels for facial recognition.

## Requirements

Make sure you have Python installed on your machine. The project's dependencies can be installed using the `requirements.txt` file.

### Dependencies

- numpy
- opencv-python
- tensorflow
- keras

To install the dependencies, run the following command:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## How to Run

1. **System Initialization**: To start the facial recognition system, run the \`main.py\` script:

   \`\`\`bash
   python main.py
   \`\`\`

2. **Real-Time Video Capture**: The \`face_recognition.py\` script handles real-time video capture and face detection. This script is automatically called by \`main.py\`.

## Code Organization

- **Models**: The facial recognition model is loaded from the \`face_recognition_model.h5\` file. This model is used to predict the classes of detected faces.

- **Class Labels**: The class labels for facial recognition are stored in the \`class_labels.npy\` file.

- **Video Capture and Processing**: Video capture and processing are managed by the \`face_recognition.py\` script, which uses the OpenCV library to capture frames in real time and perform face detection.

## Contact

For questions or suggestions, please contact the author: Vitor Mendes (vitor.mendes@ieee.org).