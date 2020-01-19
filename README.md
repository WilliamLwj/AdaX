# AdaX: Adaptive Gradient Descent with Exponential Long Term Momery

A new adaptive optimizer that can run faster than Stochastic Gradient Descent with momentum (SGDM) and get similar performance in various computer vision and natural language processing tasks.

### This is the Pytorch Implementation.

## Usage 
### Please use AdaXW by

```python3
from AdaX import AdaXW
# suppose your DNN is named 'model'
optimizer = AdaXW(model.parameters(), lr = 0.005, weight_decay=5e-2)
```


## License
[Apache 2.0](./LICENSE)
