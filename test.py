import numpy as np  # Import numpy

a = np.array([1, 2, 3, 4, 5])
# Create an array from two to twelve, not included, with step of two
b = np.arange(2, 14, 2).reshape((2, 3))
#.reshape() reshape the array for 1*6 to 2*3
print(b)
c = np.zeros(6)  # Create an array with six floating 0.
print('\n')
print(c)
print(c.dtype)
d = np.ones(6, dtype=np.int)  # Create an array with six integer 1.
print('\n')
print(d)
print(d.dtype)
# Create an 2*3 array with elements that are randomly picked between 0 to 1.
e = np.random.random((2, 3))
print('\n')
print(e)
#f = b-a
print('\n')
print(f)

print(a, b, c, d, e, f, sep='\n')
print(f < -9)
print(f == 5)
