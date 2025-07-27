# Scratch and research for the compare coastlines project



## Overview


## Framing the problem
We have a function with constants being the 3D curve and the variables are the position and orientation of the camera (to be represented with $\vec{r}$ and $q$). The output is a 2D curve which is the projection of the 3D curve as viewed by a ideal pinhole camera. 

The output of this function is compared to a target curve which is the 2D coastline adjusted for the camera parameteres (as the picture would've been taken with a pinhole camera). 

## First (and likely naive) idea

Store curves (2D and 3D) as a list of N points. Assume the curves being compared have the same start and endpoint (this takes some manual work). The curves are then represented with functions that take inputs [0,1]. The output is a point interpolated between the stored points. Likely linear interpolation but maybe we'll do a higher order interpolation. We'll likely have these curves be classes and they will be called on the objects. 

### How do you compare these objects

What is the best way to define a cost function for these objects? 

#### Idea 1

$$
a = \frac{1}{N}
$$
$$
C = \sum_{n=0}^{N}\left( \mathcal{C}(an) - \mathcal{C}_0(an)\right)^2
$$

where N+1 points on the curve are compared


## Other things that could improve this idea
- Encoding the curves differently (constrain the point and the slope at each point (eh harder to extract from coastlines and edge detected images))
- Interpolating the curve encoding with higher order interpolation between the points (if still using the point based representation)


## Resources

https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization
https://github.com/gyroflow/lens_profiles/tree/main/GoPro (source for all the camera lens models one could ever need)
https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/fisheye.cpp#L1127