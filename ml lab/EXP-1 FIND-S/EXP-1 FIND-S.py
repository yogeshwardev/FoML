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
hypothesis = ["0"] * n
print("Initial hypothesis:", hypothesis)

for i, row in enumerate(data, 1):
    if row[-1] == "Yes":
        for j in range(n):
            if hypothesis[j] == "0":
                hypothesis[j] = row[j]
            elif hypothesis[j] != row[j]:
                hypothesis[j] = "?"
    print(f"After example {i}: {hypothesis}")

print("\nMost specific hypothesis:", hypothesis)
