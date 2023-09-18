# Task 3: Face Recognition

| Info | Description |
|:-----:|:----------:|
| Project Objective |ID Face Match|
| Assignee | Ahmed Yousri Sobhi |
| Assignee's email | [ahmedyousrisobhi@gmail.com](ahmedyousrisobhi@gmail.com) |
| Department | Data Science |

## Table of Content
- [Task 3: Face Recognition](#task-3-face-recognition)
  - [Table of Content](#table-of-content)
  - [Objective](#objective)
  - [Setup Environment](#setup-environment)
  - [Dataset Folder Structure](#dataset-folder-structure)
  - [Dataset Download](#dataset-download)
  - [Inference](#inference)
    - [Setting-up Test Images Data](#setting-up-test-images-data)
    - [Notebook Inference](#notebook-inference)
    - [Script](#script)
  - [Note](#note)

## Objective
An ID face match is a biometric authentication process that involves comparing a captured facial image with a stored reference image to determine if they represent the same individual. The primary goal of this process is to verify the identity of a person by analyzing key facial features and patterns.

In this method:
- Enrollment: A reference image of the individual's face is captured and stored in a database during the enrollment process. This image serves as the template for future comparisons.
- Verification: During the verification or authentication phase, a new facial image is captured and compared to the stored reference image. The system calculates a similarity score or confidence level based on the degree of similarity between the two images.
- Matching: If the similarity score exceeds a predefined threshold, the system considers the match successful, confirming that the individual in the captured image is the same as the person associated with the reference image.
- Rejection: If the similarity score falls below the threshold, the system rejects the match, indicating that the individual's identity cannot be verified.

ID face matching is commonly used for secure access control, identity verification in various industries, and as an additional layer of security in applications such as mobile device authentication and border control.

It's important to note that the accuracy of ID face matching systems depends on factors such as the quality of the images, the algorithms used, and environmental conditions. Proper system configuration and regular updates are essential for optimal performance and security.

## Setup Environment
Navigate to the project directory:

```bash
cd project_directory
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

In case of initial setup, create required folders directory using environment script:
```bash
python ./src/env/setup_env.py

## Project Structure
```bash
.
├── data                   # Images Dataset Directory
|   ├── test
|   ├── label
├── face_match_deepface    # Face Recognition python scripts
├── nb_workspace           # Notebooks workshop
├── report                 # Reports files
|   ├── plots              # Saved figures files
├── src                    # Source scripts files
|   ├── deepface_modified  # Modified Package of deepface.
|   ├── env                # Script file to create project directory environment
└── README.md
```

## Dataset Folder Structure
The folder structure for image recognition tasks typically consists of the following:

A root directory, which contains the following subdirectories:
- train directory: This directory contains the training images and their labels.
- test directory: This directory contains the test images.
- labels directory: This directory contains the users labeled images and the label csv file.

The labels for the images are typically stored in a CSV file. The CSV file should have two columns:
- image_name (string): The name of the image.
- label (string): The label for the image.

## Dataset Download
Current used dataset is from Kaggle: [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset?select=Celebrity+Faces+Dataset)

## Inference
To inference the face recognition script, follow the following steps:

### Setting-up Test Images Data
Place the data:
- Test images in the following location: project_directory/data/test/
- Label Data:
  - Images: located at project_directory/data/label/***.jpg
  - Label csv file: located at project_directory/data/label/label.csv

### Notebook Inference
You can explore the cell interactive mode throught the notebook at: nb_worksapce/id_face_match.ipynb

### Script
The face recognition sript is located at: face_match/face_match.py

```
python ./face_match_deepface/face_recognition.py --mode video --img_dir data/test --img_video_name Video3_chunk_8.mp4 --label_path data/label
```

## Note
In case of adding new label data users, you have to remove the representations pickle file, so the model regenerate a new vector embedding for the label images.

The file is located at: data/label/representations_arcface.pkl