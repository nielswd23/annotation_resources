import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./hyperparam_results_sample1.csv")

df["sum_accuracy"] = df["training_loss"] + df["held_out_loss"]

tolerance = 0.2  # tolerance for overfitting (i.e. difference between training and held out loss), change as appropriate
result_count = 10  # how many results you want from the minimum of held_out_loss

filtered = df[(abs(df["training_loss"] - df["held_out_loss"]) < tolerance)]
three_smallest = filtered.nsmallest(result_count, "held_out_loss")[:10]
print(three_smallest.iloc[0], three_smallest.iloc[1], three_smallest.iloc[2])  # print first three
three_smallest.to_csv("./best_hyperparams_sample1.csv")

# plt.scatter(x=df["training_loss"], y=df["held_out_loss"])
# plt.show()
