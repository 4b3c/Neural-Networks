input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

import numpy as np

print(f"The actual dot product is: {input_vector[0] + input_vector[1]}")

dot_product_1 = np.dot(input_vector, weights_1)

print(f"The dot product of weights 1 is: {dot_product_1}")

dot_product_2 = np.dot(input_vector, weights_2)

print(f"The dot product of weights 2 is: {dot_product_2}")
