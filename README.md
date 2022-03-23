# Semantic State Estimation in Cloth Manipulation Tasks

## Abstract 

Understanding  of deformable object manipulations such as textiles is a challenge due to the complexity and high dimensionality of the problem.
Particularly, the lack of a generic representation of semantic states (e.g. crumpled, diagonally folded) during a continuous manipulation process introduces an obstacle to identify the manipulation type. In this paper, we aim to solve the problem of semantic state estimation in cloth manipulation tasks.
For this purpose, we introduce a new large-scale fully-annotated RGB image dataset showing various human demonstrations of different complicated cloth manipulations. 
We provide a set of baseline deep networks and benchmark them on the problem of semantic state estimation using our proposed dataset.
Furthermore, we investigate the scalability of our semantic state estimation framework in robot monitoring tasks of long and complex cloth manipulations.

## Short task dataset

The dataset is consisted of 18 different textile from which we generate 10 sematic states through 7 manipulation tasks

### Examples 
![Example Gif](/images/datagen_example.gif)


### All textiles
![Alt text](/images/All_textile_object.png "All textile objects")

### The manipulation tasks
![Alt text](/images/States_And_Manipulations.png "States and Manipulations")

## Long manipulation tasks dataset

The dataset consists 4 long manipulation tasks through human demonstration. Further 2 of those tasks are also performed by robotic execution

### Manipulation Graph
![Alt text](/images/Graph_V3.png "Manipulation Graph")

### Monitoring and class activation heatmaps

![Alt text](/images/human_robot.png "Monitoring and Class Activation Heatmaps")

## Dataset Download

We aim to provide a public download link to the dataset as soon as possible.
However, due to its large size it might take sometime to be done. 
Thus, until this issue is resolved, to access the dataset please message me through this page or by email(@gtzelepis@iri.upc.edu) and I will provide you with a temporary google drive link.

### Citation

```
@misc{https://doi.org/10.48550/arxiv.2203.11647,
  doi = {10.48550/ARXIV.2203.11647},
  
  url = {https://arxiv.org/abs/2203.11647},
  
  author = {Tzelepis, Georgies and Aksoy, Eren Erdal and Borràs, Júlia and Alenyà, Guillem},
  
  keywords = {Robotics (cs.RO), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Semantic State Estimation in Cloth Manipulation Tasks},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
