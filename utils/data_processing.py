from typing import Optional

import re
import pandas as pd


def load_df(df_path: str) -> pd.DataFrame:
    return pd.read_csv(df_path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df_no_target = df.drop(columns=['selling_price'])
    df_result = df.drop_duplicates(subset=df_no_target.columns, keep='first')
    return df_result.reset_index(drop=True)


def convert_column(df: pd.DataFrame, column_name: str, unit: str) -> pd.DataFrame:
    df[column_name] = df[column_name].astype(str).str.replace(unit, '')
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce', downcast='float')
    return df


def parse_torque_string(torque_str: str) -> tuple[Optional[float], Optional[int]]:
    if not torque_str or pd.isna(torque_str):
        return None, None

    clean_str = torque_str.lower().replace(',', '').replace('(', ' ').replace(')', ' ')
    clean_str = re.sub(r'\s+', ' ', clean_str.strip())

    patterns = [
        r'(\d+\.?\d*)\s*(nm|kgm).*?(\d+)(?:\s*-\s*(\d+))?\s*rpm',
        r'(\d+\.?\d*)\s*@\s*(\d+).*?(nm|kgm)',
        r'(\d+\.?\d*)\s*@\s*(\d+)(?:\s*-\s*(\d+))?',
        r'^(\d+\.?\d*)\s*(nm|kgm)$',
        r'(\d+\.?\d*)[^\d]+(\d+)(?:[^\d]+(\d+))?'
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_str)
        if match:
            groups = match.groups()

            if pattern == patterns[0]:
                torque_value = float(groups[0])
                unit = groups[1]
                rpm_low = int(groups[2])
                rpm_high = int(groups[3]) if groups[3] else rpm_low

            elif pattern == patterns[1]:
                torque_value = float(groups[0])
                rpm_low = int(groups[1])
                rpm_high = rpm_low
                unit = groups[2]

            elif pattern == patterns[2]:
                torque_value = float(groups[0])
                rpm_low = int(groups[1])
                rpm_high = int(groups[2]) if groups[2] else rpm_low
                unit = 'nm'

            elif pattern == patterns[3]:
                torque_value = float(groups[0])
                unit = groups[1]
                rpm_low = rpm_high = None

            else:
                torque_value = float(groups[0])
                rpm_low = int(groups[1]) if groups[1] else None
                rpm_high = int(groups[2]) if groups[2] else rpm_low
                unit = 'nm'

            if unit == 'kgm':
                torque_value = torque_value * 9.80665

            return round(torque_value, 2), rpm_high

    return None, None


def convert_torque(df: pd.DataFrame) -> pd.DataFrame:
    for index, row in df.iterrows():
        torque_str = row['torque']
        torque_value, rpm_high = parse_torque_string(torque_str)
        df.at[index, 'torque'] = torque_value
        df.at[index, 'max_torque_rpm'] = rpm_high
    return df


def convert_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = convert_column(df, 'mileage', ' kmpl')
    df = convert_column(df, 'engine', ' CC')
    df = convert_column(df, 'max_power', ' bhp')
    df = convert_torque(df)
    return df


class NoneEncoder:

    def __init__(self) -> None:
        self.medians = {}
        self.columns = ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']

    def fit(self, df: pd.DataFrame) -> None:
        for column in self.columns:
            median_value = df[column].median()
            self.medians[column] = median_value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            median_value = self.medians.get(column)
            with pd.option_context("future.no_silent_downcasting", True):
                df[column] = df[column].fillna(median_value).infer_objects(copy=False)
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
    

class TargetEncoder:

    def __init__(
        self,
        smoothing: float = 1.0
    ) -> None:
        self.smoothing = smoothing
        self.global_mean = 0
        self.target_maps = {}

    def fit(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series
    ) -> None:
        self.global_mean = y.mean()
        
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            for col in columns:
                self.target_maps[col] = {}
                for category in X[col].unique():
                    category_mean = y[X[col] == category].mean()
                    category_count = (X[col] == category).sum()
                    smoothed_mean = (category_count * category_mean + self.smoothing * self.global_mean) / (category_count + self.smoothing)
                    self.target_maps[col][category] = smoothed_mean
        else:
            self.target_maps['_series'] = {}
            for category in X.unique():
                category_mean = y[X == category].mean()
                category_count = (X == category).sum()
                smoothed_mean = (category_count * category_mean + self.smoothing * self.global_mean) / (category_count + self.smoothing)
                self.target_maps['_series'][category] = smoothed_mean

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame()
            for col in X.columns:
                result[col] = X[col].map(lambda x: self.target_maps[col].get(x, self.global_mean))
            return result
        else:
            return X.map(lambda x: self.target_maps['_series'].get(x, self.global_mean))
    
    def fit_transform(
        self,
        X,
        y: pd.Series
    ):
        self.fit(X, y)
        return self.transform(X)
    

def add_company_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['company'] = df['name'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else 'Unknown')
    return df