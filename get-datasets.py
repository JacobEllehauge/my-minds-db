from sklearn import datasets
import polars as fpd


wine = fpd.DataFrame(datasets.load_wine().data)
wine.columns = datasets.load_wine().feature_names


wine_target = datasets.load_wine().target
wine.with_columns(fpd.Series(name="wine_target", values=wine_target))

wine.write_csv("static-datasets/wine.csv", separator=",")

print("loading wine dataset:", datasets.load_wine().DESCR)
