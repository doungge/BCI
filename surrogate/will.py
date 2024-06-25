import numpy as np
from scipy.stats import wilcoxon

model1 = np.array([
75.62,
64.28,
83.43,
95.62,
94.37,
86.56,
87.18,
93.75,
82.18,
])
model2 = np.array([
76.56,
67.14,
82.81,
93.12,
95.31,
87.18,
87.50,
93.12,
88.74,
])

print(len(model1))
print(len(model2))
stat, p = wilcoxon(model1,model2)
print('Wilcoxon signed-rank test 결과')
print('통계량: ', stat)
print('p-value: ', p)



