# Learning3D Priors with Adversarial Novel View Generation

## Abstract
The  focus  of  our  work  is  to  learn 3D  shape  priors  using  single  image  per  instance  and  its  viewpointinformation.  Deep CNNs trained naively on single image datasets produce shapes that are consistent with theinput image, but exhibit arbitrary geometries when viewed from dierent perspectives.  If multiple views areavailable, we can tackle this issue usingmulti-view consistency—we enforce that the shape predicted using oneview of the object should also explain a second view of the same underlying object. Since the network cannotaccess this second viewpoint during reconstruction, it must produce a shape that looks reasonable from allperspectives. In our work, we exploit this well-tested principle for single image datasets by explicitly generatingnovel views using adversarial training.

## Architecture
<img src="https://github.com/aartika/prgan/blob/master/images/arch.png" alt="Kitten"
	title="Architecture" width="2000" />

Our approach summarized in Figure1jointly learnsshape and novel view prediction systems, denoted asfshapeandfnovel, by enforcing geometric consistencybetween their predictions.  Concretely, given one im-age of an object we predict a corresponding shape. Inparallel,  for  the  same  image,  we  independently  pre-dict another image of the same object from a dierentviewpoint sampled randomly at runtime. Then we en-force  that  the  predicted  shape,  when  observed  fromthis sampled viewpoint, should be ‘consistent’ with thenovel view image. In summary, the working principle behind our model is a mutual supervision between shape prediction andnovel view generation based on the observation that when these two systems behave optimally, their outputsare geometrically consistent.As observed earlier, shapes produced by our3D reconstructor can be arbitrarily deformed when viewedfrom other perspectives; thus, a novel view generator trained using geometric consistency alone will add novalue to our model.  To makefnoveluseful, we reinforce it with the help of a discriminator that dierentiatesbetween generated novel view images and real images from the dataset.  The discriminator forcesfnoveltoproduce images that fit the distribution of real images; at the same time, geometric consistency loss fromfshapecompels its output to be compatible with the input image.

## Results
We use the ShapeNet1dataset to empirically validate our approach. In particular, we evaluate on the5largestcategories (Table1).  As a baseline, we implement an encoder-decoder architecture wherein the3D decoder istrained using consistency loss with the input image.  We train all models using binary silhouette images asinput, and use negative cosine distance between images as the consistency metric. Our  method  achieves  a  meanintersection-over-union(IoU) of0.477 with ground truth 3D voxels, asignificant  improvement  over  the  baseline  at0.390.Compared  to  our  predicted  shapes,  baseline  predic-tions appear severely distorted when observed from aneutral viewpoint.
<div style="text-align:center"><img src="https://github.com/aartika/prgan/blob/master/images/qualitative_results.png" alt="Kitten"
	title="Qualitative Results" width="500" /></div>
