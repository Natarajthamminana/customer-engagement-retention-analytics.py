--- Initial Dataset Sample ---
   Year  CustomerId   Surname  CreditScore Geography  Gender  Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
0  2025    15634602  Hargrave          619    France  Female   42       2       0.00              1          1               1        101348.90       1
1  2025    15647311      Hill          608     Spain  Female   41       1   83807.86              1          0               1        112542.60       0
2  2025    15619304      Onio          502    France  Female   42       8  159660.80              3          1               0        113931.60       1
3  2025    15701354      Boni          699    France  Female   39       1       0.00              2          0               0         93826.63       0
4  2025    15737888  Mitchell          850     Spain  Female   43       2  125510.80              1          1               1         79084.10       0

--- Model Evaluation Output ---
Accuracy Score: 0.6666666666666666

Confusion Matrix:
[[1 1]
 [0 1]]

--- Feature Importance for Retention Strategy ---
Balance            0.207857
Age                0.145238
CreditScore        0.126190
EstimatedSalary    0.113095
Geography          0.096429
NumOfProducts      0.092857
Tenure             0.076190
IsActiveMember     0.071429
Gender             0.046429
HasCrCard          0.024286
Year               0.000000
dtype: float64
