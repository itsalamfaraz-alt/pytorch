import torch

a = torch.arange(1,13)
print("a: ",a)
print("shape of a: ", a.shape)

print("_"*40)

## we can use the view method to reshape the tensor

b = a.view(3,4)
print("b: ",b)
print("shape of b: ", b.shape)

print("_"*40)

## we can also use the reshape method to reshape the tensor but it does not require the tensor to be contiguous in memory

c = a.reshape(2, 6)
print("c: ",c)
print("shape of c: ", c.shape)

print("_"*40)

## we can also use the -1 to let the pytorch automaically calculate the size of dimension

d = a.view(-1, 4)
print("d: ",d)
print("shape of d: ",d.shape)

print("_"*40)

## PERMUTE and TRANSPOSE 

e = torch.arange(1,13).view(3,4)
print(e)
print("shape of e: ", e.shape)

print("_"*40)

f = e.transpose(0,1)
print("f: ",f)
print("shape of f: ", f.shape)

g = e.T
print("g: ",g)
print("shape of g: ", g.shape)

h = torch.permute(e,(1,0))
print("h: ",h)
print("shape of h: ", h.shape)

print("_"*40)

## indexing and slicing

i = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

print("i: ",i)
print("value at index (0,1): ", i[0,1])
print("first row: ",i[0])
print("first column: ",i[:,0])
print("last row: ",i[-1])
print("submatrix: ",i[:2,1:3]) ## it is of shape (2,2)
print("shape of submatrix: ",i[:2,1:3].shape)
print("elements greater than 5: ",i[i>5])
print("specific rows like row 0 and 2: ", i[[0,2]])
print(i[1,1]) 