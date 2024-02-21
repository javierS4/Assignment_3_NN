import numpy as  np 
num_features = 8
num_classes = 10
weights = np.random.uniform(-0.05, 0.05, (num_classes, num_features))
weights1 = np.random.uniform(-0.05, 0.05, num_features)
print(f"Weights with classes {weights}")
print(f"weights with no classes {weights1}")