import csv, io

CSV_TEXT = """Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
"""

data = list(csv.reader(io.StringIO(CSV_TEXT)))
header, data = data[0], data[1:]
n = len(header) - 1

domains = [sorted({r[i] for r in data}) for i in range(n)]

S = ["0"] * n
G = [["?"] * n]


def consistent(h, x):
    return all(h[i] == "?" or h[i] == x[i] for i in range(n))


def more_general(h1, h2):
    return all(a == "?" or a == b or b == "0" for a, b in zip(h1, h2))


for t, row in enumerate(data, 1):
    x, label = row[:-1], row[-1]
    if label == "Yes":
        G = [g for g in G if consistent(g, x)]
        for i in range(n):
            if S[i] == "0":
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = "?"
    else:
        newG = []
        for g in G:
            if not consistent(g, x):
                newG.append(g)
                continue
            for i in range(n):
                if g[i] == "?":
                    for v in domains[i]:
                        if v != x[i]:
                            h = g.copy()
                            h[i] = v
                            if more_general(h, S):
                                newG.append(h)
        G = [g for g in newG
             if not any(g != o and more_general(o, g) for o in newG)]
        G = [list(t) for t in {tuple(g) for g in G}]
    print(f"Step {t}\n  S = {S}\n  G = {G}")

print("\nFinal Specific boundary S:", S)
print("Final General boundary  G:", G)
