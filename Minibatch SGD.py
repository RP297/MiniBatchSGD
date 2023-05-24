import numpy as np

class LinearRegression:
    def __init__(self, k, eta, iterations):
        self.A = np.random.rand(100, 10)
        self.b = np.random.rand(100)
        self.x = np.zeros(self.A.shape[1])
        self.k = k
        self.eta = eta
        self.iterations = iterations

    def MiniBatchSGD(self):
        for i in range(self.iterations):
            indices = np.random.choice(self.A.shape[0], self.k, replace=False)
            A_batch = self.A[indices]
            b_batch = self.b[indices]
            gradient = 2 * np.dot(A_batch.T, np.dot(A_batch, self.x) - b_batch)
            self.x -= self.eta * gradient

    def computed(self):
        comp_obj_value = np.linalg.norm(np.dot(self.A, self.x) - self.b)**2
        return self.x, comp_obj_value

    def optimal(self):
        opt_x = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
        opt_obj_value = np.linalg.norm(np.dot(self.A, opt_x) - self.b)**2
        return opt_x, opt_obj_value

model = LinearRegression(k=10, eta=0.001, iterations=1000)
model.MiniBatchSGD()

computed_x, objective_value = model.computed()
optimal_x, optimal_objective_value = model.optimal()

print("Solution:")
print(computed_x)
print("Optimal Solution:")
print(optimal_x)
print("-"*80)
print("Objective Value (SGD):", objective_value)
print("Optimal Objective Value:", optimal_objective_value)