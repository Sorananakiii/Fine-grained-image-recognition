# Fine-grained-image-recognition
---
This repo contains training code for Image classification of the 4 fine-grained image dataset using TensorFlow. 
This implementation achieves comparable state-of-the-art results.

Fine-grained-image-recognition is a classification task where sub-category visual differences are small and can be overwhelmed by factors such as pose, viewpoint, or location of the object in the image e.g., species of birds or models of cars. The small inter-class variations and the large intra-class variations caused by the fine-grained nature makes it a challenging problem. During the booming of deep learning, recent years have witnessed remarkablen progress of FGIA using deep learning techniques.

# Dataset
---
|Dataset	      |Num Classes	|Avg samples Per Class	
|Flowers-102	  |102	        |10	
|CUB-200-2011	  |200	        |29.97	
|Stanford Cars [Download Stanford Car Dataset here]	|196	        |41.55	
|Aircrafts	    |100	        |100	
|Stanford Dogs	  |120	        |100	
1.FGVC Aircraft [Maji et al., 2013] [Download Stanford Car Dataset here]
  
2.Stanford Car [Krause et al., 2013]

3.Stanford Dog [Khosla et al., 2011]

4.CUB200-2011 [Wah et al., 2011]


  
Starting with image classification basic workflow
  1. Transfer learning with ResNet101
  
  
# Reference
---
https://arxiv.org/abs/1907.03069
  
[Download Stanford Car Dataset here]: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
