# Probabilistic convolution and deconvolution

The library implements probabilistic convolution and deconvolution on a sequence of independent random counters.



## Sequence Of Independent Counters

Let $\{x_0,x_1,\ldots,x_{N-1}\}$ be a sequence of $N$ numbers, referred to as dense counters, that are generated from independent random variables each attaining value from a finite set $\{0,1,\ldots,X\}$, however, each having a different distribution $p_i(x)$, $i=0,\ldots,N-1$. Let $P\in[0,1]^{(X+1)\times N}$ be a left stochastic matrix whose columns represent the distributions $p_i(x)$, $i=0,1,\ldots,N-1$.

Let $W$ be a positive integer not higher than $N$. Let $\{y_0,y_1,\ldots,y_{N-W}\}$ be a sequence of $N-W+1$ numbers, referred to as sparse counters, that are related to dense counters by a set of linear equations 

$$ y_i = \sum_{k=i}^{i+W-1} x_k\,, \qquad i=0,\ldots,N-W\:. $$ 

Therefore the sparse counters are numbers generated from random variables each attaining value from finite set $\{0,1,\ldots,Y\}$, where $Y=X\cdot W$, and having distributions $q_i(y)$, $i=0,\ldots,N-W$. Let $Q\in[0,1]^{(Y+1)\times (N-W+1)}$ be a left stochastic matrix whose columns represent the distributions $q_i(y)$, $i=0,1,\ldots,N-W$.

## Probabilistic convolution

Given distributions of dense counters $P\in[0,1]^{(X+1)\times N}$, the distributions of the sparse counters $Q\in[0,1]^{(Y+1)\times (N-W+1)}$ can be computed by 

$$ q_i(y) = \sum_{x_i=0}^X \sum_{x_{i+1}=0}^X\cdots\sum_{x_{i+W-1}=0}^X p_i(x_i)p_{i+1}(x_{i+1})\cdots p_{i+W-1}(x_{i+W-1}) \delta(x_i+x_{i+1}+\cdots +x_{i+W-1}=y) $$

where $\delta(A)=1$ if $A$ is true and $0$ otherwise. The operation is referred to as the probabilistic convolution with window size $W$.

## Probabilistic deconvolution

Given a left stochastic matrix $Q\in [0,1]^{(Y+1)\times (N-W+1)}$, the tasks is to find distributions of dense counters $\hat{P}\in[0,1]^{(X+1)\times N}$ such that the distribution of sparse counters $\hat{Q}$ computed from $\hat{P}$ by the probabilistic deconvolution minimizes the Kullback-Leibler divergence

$$
KL(Q|| \hat{Q}) = -\sum_{i=0}^{N-W+1} \sum_{y=0}^Y q_i(y) \log \frac{q_i(y)}{\hat{q}_i(y)}
$$

The operation is referred to as the probabilistic deconvolution with window size $W$. The probabilistic deconvolution tries to reconstruct the distribution of the dense counters $P$ from the distribution of the sparse counters $Q$. In contrast to the probabilistic convolution, the probabilistic deconvolution is not guarantted to have unique solution, i.e. there can be more distributions of dense counters which yiled the same distribution of sparse counters. 

This library computes the probabistic deconvolution by an iterative method resembling the Expectation-Maximization algorithm. The method produces a sequence $\hat{P}_0,\ldots,\hat{P}_T$ such that $\hat{Q}_0,\ldots,\hat{Q}_T$ obtained by the probabilistic convolution monotonically increases $KL(Q|| \hat{Q})$. The initial $\hat{P}_0$ is generated uniformy at random. It uses a fixed number of iterations $T$ which is set to $50$ by default. The computational and memory complexity grows linearly with $N$ but exponentially with $W$.


## Requirements

```
numpy
tqdm
matplotlib
```

