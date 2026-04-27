import numpy as np
from collections import Counter
import math

labels = np.load("clusters.npy")

counts = Counter(labels)

N = sum(counts.values())

entropy = 0

for c in counts.values():

    p = c / N

    entropy -= p * math.log2(p)

print("Estimated entropy:", entropy)
print("Clusters:", len(counts))