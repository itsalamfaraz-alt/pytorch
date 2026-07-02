import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2 + 2 * x + 1

y.backward()
print(x.grad)
print("--"*40)


## now checking if we perform two times gradient calculation, the gradient will accumulate

a = torch.tensor(3.0, requires_grad= True)
b = a * 3
b.backward()
print("gradient after first backward: ", a.grad)

a.grad.zero_() # Reset the gradient to zero before the second backward pass

c = a * 2
c.backward()
print("gradient after second backward: ", a.grad)
print("--"*40)

## now we will try detach() and no_grad() to avoid gradient accumulation

d = torch.tensor(4.0, requires_grad=True)
e = d ** 2

f = e.detach()  # Detach e from the computation graph
try:
    f.backward()  # This will not compute gradients for d
except RuntimeError as err:
    print("Error during backward on detached tensor: ", err)
print("gradient after detach: ", d.grad)

## when we use no_grad(), it will not track the gradient inside the block
with torch.no_grad():
    g = d ** 3
    
    print(g.requires_grad)
print("gradient after no_grad block: ", d.grad)
