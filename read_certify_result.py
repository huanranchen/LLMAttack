import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()
with open(args.path, "r") as json_file:
    data = json.load(json_file)

# Step 1. Histogram of certified radius
radii = [item["radius"] for item in data]
radii.sort()
print("radii are: ", radii)
print("average certified radii: ", sum(radii) / len(radii))
plt.hist(radii, bins=10, edgecolor="black")
plt.title("Radii of Certified Robustness")
plt.xlabel("Certified Radius")
plt.ylabel("Frequency")
plt.show()
print("-"*100)

# Step 2. Histogram of pA
pAs = [item["pA"] for item in data]
pAs.sort()
print("pAs are: ", pAs)
print("average pA: ", sum(pAs) / len(pAs))
plt.hist(pAs, bins=30, edgecolor="black")
plt.title("Distribution of pA")
plt.xlabel("pA")
plt.ylabel("Frequency")
plt.show()
print("-"*100)

# Last Step. Visualize all stored values.
for data_each_input in data:
    print(data_each_input)
    print("-" * 100)
