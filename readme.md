# Anomaly Detection
This reposiory contains code that implements an anomaly detector.
The anomaly detector is based on the formulation in the paper "Estimating the support of a high-dimensional distribution", by Scholkopf, Platt, Shawe-Taylor, Smola, Williamson.

Even though I follow the formulation in the paper, the algorithm is an adaptation of the Douglas-Rachford algorithm, the details of which I'll desribe elsewhere.

There is one script, namely `flower.py` that produces the figures in this readme file, so as to demonstrate the use of the code. You can directly run `flower.py`,
as

    ./flower.py
from under the directory `./code/` -- it saves some figures under `./figures/`. You can also specify some arguments when running `./flower.py`. Try

    ./flower.py -h

to see your options. 

Below, I briefly describe what the detector is doing.

## Estimating the Support of a Flower
Suppose we are given a set of points $x_1,\ldots, x_K$, with $x_i \in \mathbb{R}^2$ as shown below.
![](./figures/flower.png)

The support estimation scheme constructs a function of the form
$$
f_{\alpha}(x) = \sum_{i=1}^K\,\alpha_i\,K(x_i, x),
$$
where
$$
K(x,t) = \exp\left(\frac{\|x - t\|_2^2}{\sigma^2}\right),
$$
and $\alpha_i$'s are determined by solving a convex minimization problem.
After solving the minimization problem, a significant number of $\alpha_i$'s are zero. Note that if $\alpha_i=0$, then the data point $x_i$ has no  effect on $f_{\alpha}(\cdot)$.

Below, I show the points $x_i$ for which $\alpha_i>0$, where the color indicates the magnitude of $\alpha_i$. Notice that the number of "support points" is in fact significantly lower than the number of input data points. Notice also that the points are selected so as to lie on the boundary of the flower.

![](./figures/nonzero.png)

The resulting function $f_{\alpha}(\cdot)$ is shown below. 
![](./figures/function.png)

Even though the support points lie on the boundary of the flower, the decision function is fairly constant inside the flower. That behavior depends on the value of $\sigma^2$ in the definition of the kernel. With a smaller $\sigma^2$, we expect sharper boundaries, and many more support points inside the flower.

In order to experiment with that parameter, you can try, for instance

    ./flower.py -v 3e-2

*Ilker Bayram, ibayram@ieee.org, 2021*