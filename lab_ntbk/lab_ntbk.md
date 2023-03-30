# March 30th, 2023

### To-Do: (things I need to change)
- :rotating_light: Getting `NaN` error when using Riemanian ADAM optimizer from `geoopt` -- however I suspect there may be a mistake in rADAM implementation. Therefore, I am temporarily switching to Riemanian SGD to see if that removes the error!
- Investigate if they forgot to square the gradient-norm in line $93$ of the `radam.py` file in `geoopt`.
- Investigate if the check that a point is containted on a manifold has bad value clipping such that it is possible to get a `NaN` error from the sqrt function


### Done: (things I've actually changed)
- In Riemannian SGD blog post (linked below) they give an example implementation with `geoopt`. They make a few addtions which I've added:
    - :rotating_light: Before passing in $X$ as a manifold parameter, they first make $X$ a manifold tensor.
    - They pass in a parameter to the Riemannian ADAM algorithm that *stablizes* it every $n$ iterations. They set $n=1$ and I do as well.
- Investigate why `Likelihood.parameters()` has two parameters to optimize: the hyperbolic embeddings $X$ and the manifold curvature parameter $\rho$. The *only* parameter should be $X$!
  - Lorentz manifold, by default, makes the curvature $\rho$ a Manifold parameter but sets `requires_grad=False`. I fixed this by directly passing in `[l.X]` -- a list of the hyperbolic embeddings $X$ -- to our optimizer


### Notes: (thoughts / conceptual understanding)
- Investigated blog post that explains **Riemanian SGD** really well [here](https://andbloch.github.io/Stochastic-Gradient-Descent-on-Riemannian-Manifolds/). They describe the algorithm as follows
    1. Compute gradient $h := \nabla_\theta \mathcal{L}(\theta^t) = g^{-1}_{\theta^t} \frac{\partial \mathcal{L}(\theta^t)} {\partial \theta}$ for loss function $\mathcal{L}$, normal auto-grad derivatives $\frac{\partial \mathcal{L}(\theta^t)} {\partial \theta}$, and the inverse metric tensor $g_{\theta^t}^{-1}$
    2. Project gradient $h$ onto tangent space $\mathcal{T}_{\theta^t}\mathcal{M}$ to get vector $v = \textrm{proj}_{\mathcal{T}_{\theta^t}\mathcal{M}} h$
    3. Perform optimizer update. With SGD, we compute $ -\eta v$ which still lies on the tangent plane $\mathcal{T}_{\theta^t}\mathcal{M}$. With ADAM, this update is different.
    4. Project back onto manifold $\mathcal{M}$ through exponential map or retraction:
        - $\theta^{t+1} = \textrm{exp}( -\eta v)$. Exponential map involves computing difficult-to-compute differential equation
        - $\theta^{t+1} = \textrm{proj}_{\mathcal{M}} (\theta^t - \eta v )$. Retraction is a first order approximation to the exponential map.

# March 3rd, 2023

### To-Do:
- Check that the gradient provided by `geoopt` is the same gradient that I would get when manually computing it
- Switch away from using `geoopt` and instead manually write code in pytorch. Then compute the gradient, only using the Wilson convention.

### Done:
- Confirmed the proper hyperboloic notation (in terms of $+$ and $-$ signs for inner product definition)
- When checking that a point is on the hyperboloid, must check two requirements:
    - $<x,x> = - \rho^2$
    - $x[0] > 0$

  Note that the inner product $<\cdot, \cdot>$ and indexing the first element $x[0]$ are both assuming the geoopt convention for the hyperboloid model. I specifically added the second requirement to the `contains()` function when checking that a point lies in the `geoopt` hyperboloid model.

### Questions:

### Notes:
- Investigating why `geoopt`'s gradient updates slowly make points drift off the hyperboloid model. Two leading hypothesis:
    - Precision error
    - Implementation mistake

  Can see points drifting off hyperboloid: ![image](images/drifting_off_hyperboloid.png "Drifting off hyperboloid")

  One can observe that the negative curvature squared, $- \rho^2$ is equal to $-4$ as given by the variable `rho1`. The single data point, given by `mdp`, 'wobbles' around the value of $-4$ before drifting to far away and triggering an assertion error.