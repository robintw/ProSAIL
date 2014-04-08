# Import the model class
from prosail import Prosail

p = Prosail()

results = p.run(1.5, 40, 8, 0, 0.01, 0.009, 1, 3, 0.01, 30, 10, 10, p.Planophile)
print results

# Results ready for use with Py6S
results2 = p.run(1.5, 40, 8, 0, 0.01, 0.009, 1, 3, 0.01, 30, 10, 10, p.Planophile, Py6S=True)
print results2

# Use these results with Py6S by running something like:
# s = SixS()
# s.ground_reflectance = GroundReflectance.HomogeneousLambertian(results2)
# s.run()