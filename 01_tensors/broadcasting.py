import torch

## BROADCASTING = arithmatic operations on different shapes of tensors

a = torch.tensor([[1,2,3],[4,5,6]])

b = torch.tensor([1,2,3])

## pytorch will automatically expand the smaller tensor into the shape of the larger tensor to perform the operations

c = a + b
print ("c: ", c)
print("shape of c: ",c.shape)

print("_"*40)

## IN-PLACE OPERATIONS = operations that modify the original tensor (_ suffix is used to denote in-place operations)

x = torch.tensor([1,2,3])
print("before: ",x)

x.add_(10)
print("after: ",x)

x.mul_(2)
print("after: ",x)

## dont use in-place operations on tensors that require gradients as it will break the computation graph and result in an error during backpropagation

y = torch.tensor([1.0,2.0,3.0], requires_grad=True)
try:
    y.add_(10)
except RuntimeError as e:
    print("Error: ", e)