import torch
a = torch.tensor([2.0, 4.0])
b = torch.tensor([1.0, 3.0])

suma = a + b
producto = a * b

print("Suma:", suma)
print("Producto:", producto)

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 5  # Funcion a sacar la derivada
y.backward()  # Calcula el gradiente dy/dx

print(x.grad)  # Resultado de la derivada ya evaluada (en 2)