import torch

# creating a tensor with pytorch

a = torch.tensor([1,2,3])
print("a: ", a)
print("dtype: ", a.dtype)
print("="*40)

# creating a tensor of zeros od shaoe (2,3)

b = torch.zeros(2,3)
print("b: ", b)
print("dtype: ", b.dtype)
print("="*40)

# creating a tensor of ones of shape (2,3)

c = torch.ones(2,3)
print("c: ", c)
print("dtype: ", c.dtype)
print("="*40)

# creating a tensor of random numbers of shape (2,3)

d = torch.randn(2,3)
print("d: ", d)
print("dtype: ", d.dtype)
print("="*40)

# creating a tensor with a range of values

e = torch.arange(0, 10, 2)
print("e: ", e)
print("dtype: ", e.dtype)
print("="*40)

# creating a tensor with a specific shape and data type

f = torch.empty(2, 3, dtype=torch.float32)
print("f: ", f)
print("dtype: ", f.dtype)
print("="*40)

# checking shape and device 

print("shape of a : ", a.shape)
print("device of a : ", a.device)
print("="*40)

# changing the data type of tensor "a"

g = a.to(torch.float32)  # we can also use a.float() to change the dtype
print("g: ", g.dtype)
print("="*40)

# checking if CUDA is available

print("CUDA availability: ", torch.cuda.is_available())
print("="*40)

# setting the device to GPU if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

h = torch.tensor([1,2,4], device=device)
print("h: ",h)
print(h.device)
