## Probabilistic convolution and deconvolution

The library implements probabilistic convolution and deconvolution on a sequence of independent random counters.

The exact formulation of the problem and an example is in the Jupyter botebook
```
pdeconv_example.ipynb
```

$$x^2$$

$$ y_i = \sum_{k=i}^{i+W-1} x_k\,, \qquad i=0,\ldots,N-W\:. $$ 


### Requirements

```
numpy
tqdm
matplotlib
```

