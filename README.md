# PyTorchOT

Implements sinkhorn optimal transport algorithms in PyTorch. Currently there are two versions of the Sinkhorn algorithm implemented: [the original](https://arxiv.org/pdf/1306.0895.pdf) and the [log-stabilized version](https://arxiv.org/pdf/1610.06519.pdf). This code essentially just reworks a couple of the implementations from the awesome [POT library](https://github.com/rflamary/POT/) in PyTorch.

Example usage:
```python
from ot_pytorch import pairwise_distance_matrix, sink

M = pairwise_distance_matrix(x, y)
dist = sink(M, reg=5)
```

The examples.py file contains three basic examples.

### Example 1

The algorithm yields:

![alt text](https://github.com/vincentqb/PyTorchOT/blob/master/plots/uniform_example/uniform_stabilized_example1.png)

### Example 2

Let Z<sub>i</sub> ~ Uniform[0,1], and define the data X<sub>i</sub> = (0,Z<sub>i</sub>), Y<sub>i</sub> = (θ, Z<sub>i</sub>), for i=1,...,N and some parameter θ which is varied over [-1,1]. The true optimal transport distance is |θ|. The algorithm yields:

![alt text](https://github.com/vincentqb/PyTorchOT/blob/master/plots/uniform_example/uniform_example2.png)

### Example 3

The algorithm yields:

![alt text](https://github.com/vincentqb/PyTorchOT/blob/master/plots/gaussian_example/gaussian_example3.png)
