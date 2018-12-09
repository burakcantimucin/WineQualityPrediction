# coding: utf-8
import csv

def main():
    with open("red-wine-quality-prediction.csv") as myfile:
        data = csv.DictReader(myfile, delimiter=',')
        list = []
        for row in data:
            list.append({"fixed-acidity":row["fixed acidity"],
                         "volatile-acidity":row["volatile acidity"],
                         "citric-acid":row["citric acid"],
                         "residual-sugar":row["residual sugar"],
                         "chlorides":round(float(row["chlorides"]), 3),
                         "free-sulfur-dioxide":row["free sulfur dioxide"],
                         "total-sulfur-dioxide":row["total sulfur dioxide"],
                         "density":round(float(row["density"]), 3),
                         "pH":row["pH"],
                         "sulphates":row["sulphates"],
                         "alcohol":row["alcohol"],
                         "real-quality":row["real quality"],
                         "predicted-quality":row["predicted quality"]})
    return list

if __name__ == "__main__":
    main()