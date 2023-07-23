# Project Trinetra

Project Trinetra is an early-stage research project inspired by Google's Guideline project. Its primary objective is to enable individuals with low vision to run independently using computer vision and machine learning. The core technology utilized in Trinetra is a semantic segmentation model that classifies every pixel in the frame as "walkable" or "not walkable" to predict the runner's position relative to the safe path.

## System Architecture

The Trinetra system comprises the following key components:

- **Client Phone**: This is the mobile device equipped with a camera that captures live frames of the surroundings.
- **Local Server**: The AI model is hosted on a local server, which processes the captured frames and performs semantic segmentation.

The semantic segmentation model used in Trinetra is based on the well-known U-Net architecture, which has been adapted to generate grayscale masks representing the confidence of each pixel's prediction. These masks are used to determine the areas that are safe for the runner to walk on.

## Challenges Addressed

Trinetra addresses the following technical challenges:

- **System Accuracy**: To ensure user safety, the segmentation model must accurately and reliably predict the runner's position relative to the walkable path in various environments and conditions.
- **System Performance**: The system needs to process frames at a minimum of 15 frames per second to provide real-time feedback to the user. It must also be able to run for extended periods (at least 3 hours) without significantly draining the phone's battery. Furthermore, the system should work offline without relying on an internet connection, ensuring its usability in areas with limited or no data service.

## Project Description

Trinetra leverages semantic segmentation to predict walkable areas in the frames captured by the client phone's camera. The core workflow involves the following steps:

1. **Semantic Segmentation**: The frames are fed into the semantic segmentation model, which generates grayscale masks. These masks distinguish between areas considered safe for walking (walkable) and those that are not safe (not walkable).

2. **Path Planning Algorithms**: The masked frames with identified walkable areas are then processed using path planning algorithms to determine the safest and most efficient path for the runner.

   a. **Algorithm 1**: The initial approach involves dividing the masked frame into k parts and using the centroid of each part to predict the safe walkable path.

   b. **Algorithm 2**: In this approach, a backtracking algorithm is employed to find the walkable path, exploring various options to reach the destination safely.

   c. **Algorithm 3**: An enhanced version of Algorithm 2 is implemented to make it more adaptable and optimize it for the specific problem statement.

## Conclusion

Project Trinetra represents a promising effort in enabling individuals with low vision to run independently with the aid of computer vision and machine learning technologies. The team is actively working on improving the system's accuracy and performance to provide reliable and real-time feedback to users. Trinetra aims to empower those with low vision to enjoy a more independent and fulfilling running experience while ensuring their safety in different environments and conditions.
