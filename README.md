# Fine-grained-image-recognition
---
This repo contains training code for Image classification of the 5 fine-grained image dataset using TensorFlow. 
This implementation achieves comparable state-of-the-art results.

FVGR is a classification task where intra category visual differences are small and can be overwhelmed by factors such as pose, viewpoint, or location of the object in the image. For instance, the following image shows a California gull (left) and a Ringed-beak gull (Right). The beak pattern difference is the key for a correct classification. Such a difference is tiny when compared to the intra-category variations like pose and illumination.

# Dataset
---

1.FGVC Aircraft [Maji et al., 2013] [Download Stanford Car Dataset here]
  
2.Stanford Car [Krause et al., 2013]

3.Stanford Dog [Khosla et al., 2011]

4.CUB200-2011 [Wah et al., 2011]

5.Oxford Flower [Nilsback and Zisserman, 2008

  
Starting with image classification basic workflow
  1. Transfer learning with ResNet101
  
  
  
  
[Download Stanford Car Dataset here]: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
