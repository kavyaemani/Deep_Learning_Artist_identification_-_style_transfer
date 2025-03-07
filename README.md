# Deep Learning Artist Identification & Style Transfer(Code was run using Northwestern MLDS' GPU cluster.)
This project develops an end-to-end deep learning pipeline to automatically identify artists from their paintings and to perform art style transfer. Two main tasks are addressed:

- Artist Identification: Classify paintings by 11 top artists (filtered from a dataset of 50) using a CNN-based approach.
- Art Style Transfer: Transfer the style of one painting to another image by optimizing content and style losses.

# Artist Identification
- Data Processing
    - Dataset Filtering: Although the dataset includes paintings by 50 artists, only 11 artists have more than 200 paintings. To reduce computation and improve training, only these 11 artists are used.
    - Handling Imbalance:
    - The dataset is imbalanced (e.g., Van Gogh has 877 paintings vs. Marc Chagall’s 239). Class weights are computed and applied during training, which substantially improved performance.
- Data Augmentation:
  - Keras’ ImageDataGenerator is used to augment data (shear, horizontal & vertical flips) carefully, since this is a style identification task rather than traditional object detection.
- Modeling & Training
  Architecture: A CNN-based approach is adopted using a pre-defined architecture. After testing multiple models, ResNet50 (with pretrained ImageNet weights) worked best.
- Fine-Tuning Strategy: The model is designed to focus on the artistic style rather than objects. Therefore, more emphasis is placed on training shallow layers while freezing deeper layers.
- Training Details: The model is trained in two phases with callbacks like ReduceLROnPlateau and EarlyStopping. This approach achieved approximately 99% training accuracy and 85% cross-validation accuracy.

# Art Style Transfer
- Preprocessing: Images are resized to a consistent size (e.g., 224×224) and normalized.
- Feature Extraction: Pretrained models such as VGG19 or InceptionV3 are used to extract both content and style features from images.
- Loss Computation:
  - Content Loss: Measures differences between the generated and base images.
  - Style Loss: Uses the Gram Matrix to measure the similarity of style between images.
- Optimization: The generated image is iteratively updated to minimize the combined content and style losses.
-
# Results
- Artist Identifier: The final model identified artists with about 99% accuracy on training data and 85% accuracy on cross-validation data.
- Style Transfer: The optimized images successfully combine content from one image with the style of another, demonstrating the potential for computational creativity.

# How to Use
Installation:
Clone the repository and install dependencies with:
pip install -r requirements.txt
Dataset Preparation:
Ensure that your dataset is organized with each artist’s images in a separate subdirectory. Adjust the file paths in the notebooks as needed.
Training:
Run the Deep_Learning_Artist_identification.ipynb notebook to train the artist identifier.
Style Transfer:
Follow the instructions in the Art_Style_Transfer.ipynb notebook for style transfer tasks.
Future Work
Explore further fine-tuning and advanced data augmentation techniques.
Experiment with alternative architectures like EfficientNet.
Optimize style transfer for faster convergence and higher quality outputs.
License
This project is licensed under the MIT License.
