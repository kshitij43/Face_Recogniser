# Face_Recogniser
Real-time video face recognizer using OpenFace

To run the code:

1] Copy the folder provided onto the OpenFace root directory.

2] The folder should have a folder named "training-images" (if not create one).

3] "training-images" folder should have atleast 2 folders for different persons, each having atleast 20 images of their respective persons face.

4] Run the "start.sh" file to create aligned face images in "aligned-images" and csv files containing generated labels and a trained SVM-classifier "classifier.pkl" file in "generated-embeddings" folder.

5] Now run the "face_recogniser.py" to start the webcam to recognise faces.

6] To test on an input image, run the command:
"./../demos/classifier.py infer ./generated-embeddings/classifier.pkl <Image_File>"


Software dependencies:

python 2.7.12
NumPy 1.11.2
SciPy 0.18.1
OpenCV 3.1.0
dlib
Openface
Torch

tested on UBUNTU 16.04 LTS
