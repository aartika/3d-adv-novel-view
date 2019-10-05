# Learning3D Priors with Adversarial Novel View Generation

## Abstract
The  focus  of  our  work  is  to  learn 3D  shape  priors  using  single  image  per  instance  and  its  viewpoint information.  Deep CNNs trained naively on single image datasets produce shapes that are consistent with the input image, but exhibit arbitrary geometries when viewed from different perspectives. 

If multiple views are available, we can tackle this issue using multi-view consistency. We enforce that the shape predicted using one view of the object should also explain a second view of the same underlying object. Since the network cannot access this second viewpoint during reconstruction, it must produce a shape that looks reasonable from all perspectives. 

In our work, we exploit this well-tested principle for single image datasets by explicitly generating novel views using adversarial training.

## Architecture
<img src="https://github.com/aartika/prgan/blob/master/images/arch.png" alt="Kitten"
	title="Architecture" width="2000" />

Our approach summarized in Figure 1 jointly learns shape and novel view prediction systems, denoted as <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " />  <i>fshape</i> and <i>fnovel<i>, by enforcing geometric consistency between their predictions.  Concretely, given one image of an object we predict a corresponding shape. 

In parallel,  for  the  same  image,  we  independently  predict another image of the same object from a different viewpoint sampled randomly at runtime. We then enforce  that  the  predicted  shape,  when  observed  from this sampled viewpoint, should be <i>consistent</i> with the novel view image. 

In summary, the working principle behind our model is a mutual supervision between shape prediction and novel view generation based on the observation that when these two systems behave optimally, their outputs are geometrically consistent. 

As observed earlier, shapes produced by our 3D reconstructor can be arbitrarily deformed when viewed from other perspectives; thus, a novel view generator trained using geometric consistency alone will add no value to our model. To make <i>fnovel</i> useful, we reinforce it with the help of a discriminator that differentiates between generated novel view images and real images from the dataset. The discriminator forces <i>fnovel</i> to produce images that fit the distribution of real images; at the same time, geometric consistency loss from <i>fshape</i> compels its output to be compatible with the input image.

## Results
We use the ShapeNet_1  dataset to empirically validate our approach. In particular, we evaluate on the five largest categories (Table1).  As a baseline, we implement an encoder-decoder architecture wherein the 3D decoder is trained using consistency loss with the input image.  We train all models using binary silhouette images as input, and use negative cosine distance between images as the consistency metric. 

Our  method  achieves  a  mean intersection-over-union (IoU) of 0.477 with ground truth 3D voxels, a significant improvement  over  the  baseline  at 0.390. Compared to our predicted shapes, baseline predictions appear severely distorted when observed from a neutral viewpoint.

<div style="text-align:center"><img src="https://github.com/aartika/prgan/blob/master/images/qualitative_results.png" alt="Kitten" title="Qualitative Results" width="500" /></div>
