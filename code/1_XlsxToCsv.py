import pandas as pd

input_data = pd.read_excel("input_data.xlsx", sheet_name=None)
learn_data = pd.read_excel("learn_data.xlsx", sheet_name=None)

input_data["02_대여소"].to_csv("input_data/spots.csv", index=False, encoding="utf-8")
input_data["03_거리"].to_csv("input_data/dist.csv", index=False, encoding="utf-8")
input_data["04_05_접근환경"].to_csv("input_data/nearby.csv", index=False, encoding="utf-8")
input_data["06_07_달력기상"].to_csv("input_data/calclim.csv", index=False, encoding="utf-8")

learn_data["01_시간별수요"].to_csv("learn_data/demand.csv", index=False, encoding="utf-8")
learn_data["02_대여소"].to_csv("learn_data/spots.csv", index=False, encoding="utf-8")
learn_data["03_거리"].to_csv("learn_data/dist.csv", index=False, encoding="utf-8")
learn_data["04_05_접근환경"].to_csv("learn_data/nearby.csv", index=False, encoding="utf-8")
learn_data["06_07_달력기상"].to_csv("learn_data/calclim.csv", index=False, encoding="utf-8")
