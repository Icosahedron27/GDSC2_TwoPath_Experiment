from pathlib import Path
import pandas as pd
import math
import numpy as np


def computeMeanDeviation(df: pd.DataFrame):
    data = df.iloc[1:, 1:]
    cell_line_number = data.shape[0]
    scalar = 1 / cell_line_number
    mean = []

    for column in data.columns:
        sum_z_column = 0.0
        for entry in data[column]:
            if entry == '' or pd.isna(entry):
                continue
            sum_z_column += float(entry)
        mean.append(scalar * sum_z_column)

    deviation = []
    for i, column in enumerate(data.columns):
        dev = 0.0
        for entry in data[column]:
            if entry == '' or pd.isna(entry):
                continue
            dev += (float(entry) - mean[i]) ** 2
        dev = math.sqrt(scalar * dev)
        deviation.append(dev)

    return mean, deviation


def computeZScore(mean: float, deviation: float, xValue) -> float:
    return float((xValue-mean)/deviation)

def normalizeDesignMatrix(path: Path):
    X = pd.read_parquet(path / 'X.parquet')
    y = pd.read_parquet(path / 'y.parquet')

    meansX, deviationX = computeMeanDeviation(X)
    meansY, deviationY = computeMeanDeviation(y)

    X_normalized = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    y_normalized = pd.DataFrame(0.0, index=y.index, columns=y.columns)

    i = 0
    for columnX in X.columns:
        for idx in X.index:
            originalXValue = X.at[idx, columnX]
            X_normalized.at[idx, columnX] = computeZScore(meansX[i], deviationX[i], originalXValue)
        i += 1

    l = 0
    for columnY in y.columns:
        for idy in y.index:
            originalYValue = y.at[idy, columnY]
            y_normalized.at[idy, columnY] = computeZScore(meansY[l], deviationY[l], originalYValue)
        l += 1
    
    X_normalized.to_parquet("X_zScoreNormalized.parquet")
    y_normalized.to_parquet("Y_zScoreNormalized.parquet")


if __name__ == "__main__":
    path = Path('data/processed/Docetaxel/v1-union-na20/design_matrix')
    normalizeDesignMatrix(path)
