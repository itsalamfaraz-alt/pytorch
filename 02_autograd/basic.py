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