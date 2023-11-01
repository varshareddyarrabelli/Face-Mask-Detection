# Face Mask Detection

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Features](#features)
- [Applications](#applications)

## Project Overview

The "Face Mask Detection" project is a real-time computer vision system designed to detect and identify individuals who are not wearing face masks. The system uses deep learning techniques, specifically the MobileNetV2 architecture, and is implemented using popular libraries and frameworks such as Keras, scikit-learn, TensorFlow, and OpenCV. The primary motivation for this project is to enhance safety and public health, particularly in the context of the COVID-19 pandemic. This project was trained using 1900 images. Dataset used in this project are used from kaggle, google images and few open source image libraries. It offers a practical and efficient solution to ensure adherence to face mask regulations in various settings, including airports, railway stations, offices, schools, and public places.

## Installation

1. Download the zip folder.
2. Extract the folder.
3. open train_mask.py and change the DIRECTORY variable to your current directory.
4. Open Command Promt and run this command:

        pip install -r installations.txt

5. Run the below command for training the data to the model

        python train_mask.py

6. Now, by running the below command, you can run the model and check whether the face is masked or not.

        python detect_mask.py

## Features

* Face Detection: The system employs a pre-trained face detection model to identify individuals within the video frame.

* Mask Classification: A pre-trained mask detection model based on MobileNetV2 architecture is used to classify individuals into two categories: wearing masks and not wearing masks.

* Real-Time Alerts: When a person is detected without a mask, the system provides real-time alerts, such as visual indicators or notifications, to ensure immediate intervention.

* Efficiency: The use of MobileNetV2 allows the model to run efficiently on resource-constrained hardware, making it suitable for real-world deployment.

* Data Visualization: The project includes a data visualization component that displays the video feed with percentage of wearing a mask, making it easy for users to interpret the results.

## Applications

Some of the Applications, where face mask detection can be used are:
 
* Public Places: Airports, railway stations, bus terminals, shopping malls, and restaurants can deploy this system to enforce mask-wearing guidelines.

* Workplaces: Offices, factories, and warehouses can use the system to ensure employee safety and compliance with mask mandates.

* Educational Institutions: Schools, colleges, and universities can use the system to monitor students and staff for safety.

* Healthcare Facilities: Hospitals and clinics can enhance patient and healthcare worker safety through continuous monitoring.

* Government Buildings: Government offices and public institutions can implement the system for public safety compliance.