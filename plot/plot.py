import pandas as pd
import matplotlib.pyplot as plt
def load_index_value_file(path):
    xs = []
    ys = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, v = line.split(":")
            xs.append(int(i))
            ys.append(float(v))

    return pd.DataFrame({"x": xs, "y": ys})


df1 = load_index_value_file("finese_any_gen.txt")

df2 = load_index_value_file("finese_any_gen_exact.txt")





plt.plot(df1["x"], df1["y"], label="NN + GA Glouton")
plt.plot(df2["x"], df2["y"], label="NN + GA PLNE")

plt.xlabel("Nombre de générations")
plt.ylabel("Valeur de la fonction objectif du KCTSP")
plt.legend()
plt.grid(True)
plt.show()
