from typing import List, Dict, Any
# Modules you can use (not all are mandatory):
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error)
import random
import polars as pl  # Polars should be used for data handling.
import csv
import os
from sklearn.exceptions import NotFittedError


class HousePricePredictor:

    train_data: pl.DataFrame
    test_data: pl.DataFrame

    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.

        Args:
            train_data_path (str): Path to the training dataset CSV file.
            test_data_path (str): Path to the testing dataset CSV file.

        Attributes to Initialize:
            - self.train_data: Polars DataFrame for the training dataset.
            - self.test_data: Polars DataFrame for the testing dataset.
        """

        self.train_data = pl.read_csv(train_data_path, null_values=["NA"])
        self.test_data = pl.read_csv(test_data_path, null_values=["NA"])
        self.data = {"train": self.train_data, "test": self.test_data}

    def clean_data(self):
        """
        Perform comprehensive data cleaning on the training and testing datasets.

        Tasks:
        1. Handle Missing Values:
            - Use a strategy for each column: drop, fill with mean/median/mode, or create a separate category.
                - Strategy that we'll use:
                    - Convert categorical variables that have strings as values to integers. NAs will be dropped. 
                    (strategy will be applied unless columns are specified to have another strategy applied to them)
                    - Mean: we will use this strategy for:
                        - OverallQual, OverallCond, ExterQual, ExterCond, BsmtCond, TotalBsmtSF, FireplaceQu, GarageArea, GarageCond
                        - HeatingQC, GrLivArea, KitchenQual, WoodDeckSF, OpenPorchSF, EnclosedPorch,
                    - Assign random integer categorical value: CentralAir, PavedDrive
                    - Mode: BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars, PoolArea, SaleType, SaleCondition
                    - Other to drop:
                        - MiscVal
        2. Ensure Correct Data Types:
            - Convert numeric columns to float/int.
            - Convert categorical columns to string.
        3. Drop Unnecessary Columns:
            - Identify and remove columns with too many missing values or irrelevant information.

        Tips:
            - Use Polars for data manipulation.
            - Implement a flexible design to allow column-specific cleaning strategies.
        """

        # Convert categorical variables to integers
        categorical_columns_with_text_values = [
            "MSZoning", "LandContour", "Utilities", "LotConfig", "LandSlope",
            "Neighborhood", "Condition1", "Condition2", "BldgType",
            "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
            "Exterior2nd", "Foundation", "Heating", "GarageType",
            "MiscFeature", "MiscVal", "CentralAir", "PavedDrive", "SaleType",
            "SaleCondition"
        ]

        categorical_columns_with_quality_score_as_text = [
            "ExterQual", "ExterCond", "BsmtCond", "HeatingQC", "KitchenQual",
            "FireplaceQu"
        ]

        # Handle missing values
        columns_to_average = [
            "OverallQual",
            "OverallCond",
            "ExterQual",
            "ExterCond",
            "BsmtCond",
            "TotalBsmtSF",
            "FireplaceQu",
            "GarageArea",
            "GarageCond",
            "HeatingQC",
            "GrLivArea",
            "KitchenQual",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
        ]

        columns_to_random_integer_categorical_value = [
            "CentralAir",
            "PavedDrive",
        ]

        columns_to_mode = [
            "BsmtFullBath",
            "BsmtHalfBath",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "TotRmsAbvGrd",
            "Fireplaces",
            "GarageCars",
            "PoolArea",
            "SaleType",
            "SaleCondition",
        ]

        columns_to_drop_entirely = [
            "LotFrontage", "Street", "Alley", "LotShape", "YearBuilt",
            "YearRemodAdd", "MasVnrType", "MasVnrArea", "BsmtQual",
            "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2",
            "BsmtFinSF2", "BsmtUnfSF", "Electrical", "1stFlrSF", "2ndFlrSF",
            "LowQualFinSF", "Functional", "GarageYrBlt", "GarageFinish",
            "GarageQual", "3SsnPorch", "ScreenPorch", "PoolQC", "Fence",
            "MoSold", "YrSold"
        ]

        columns_to_drop_nas = set(categorical_columns_with_text_values) - set(
            columns_to_average)
        columns_to_drop_nas.add("MiscVal")

        for key, data_set in self.data.items():
            num_rows = data_set.height  # Total number of rows in the DataFrame
            high_na_columns = [
                col for col in data_set.columns
                if data_set.select(pl.col(col).is_null().sum()).item() / num_rows > 0.2
            ]
            columns_to_drop_entirely = [*columns_to_drop_entirely, *high_na_columns]
            
            categorical_data_with_text_values = data_set.select([
                pl.col(column)
                for column in categorical_columns_with_text_values
            ])

            categorical_mappings = {}
            for column_name in categorical_data_with_text_values.columns:
                unique_values = categorical_data_with_text_values[
                    column_name].unique()
                unique_values = [
                    value for value in unique_values if value != "NA"
                ]
                value_to_index = {}
                index = 1
                for value in unique_values:
                    if value != "NA":
                        value_to_index[value] = index
                        index += 1
                categorical_mappings[column_name] = value_to_index

            categorical_columns_with_quality_score_as_text_indices = {
                "Ex": 1,
                "Gd": 2,
                "TA": 3,
                "Fa": 4,
                "Po": 5
            }

            # Replace text values with integers
            for column_name, value_to_index in categorical_mappings.items():
                data_set = data_set.with_columns(
                    pl.col(column_name).replace(value_to_index).alias(
                        column_name)
                )

            for column_name in categorical_columns_with_quality_score_as_text:
                data_set = data_set.with_columns(
                    pl.col(column_name).replace(
                        categorical_columns_with_quality_score_as_text_indices).alias(column_name)
                )

            # Drop missing values in specific columns
            for column in columns_to_drop_nas:
                data_set = data_set.drop_nulls(column)
            # Fill with average
            for column in columns_to_average:
                data_set = data_set.with_columns(
                    pl.col(column).fill_null(pl.col(column).mean()))
            # Fill with mode
            for column in columns_to_mode:
                data_set = data_set.with_columns(
                    pl.col(column).fill_null(pl.col(column).mode()))

            # Fill with random integers for categorical variables
            if not categorical_mappings:
                raise Exception("Categorical mapping not found")
            for column in columns_to_random_integer_categorical_value:
                data_set = data_set.with_columns(
                    pl.col(column).fill_null(
                        random.choice(list(
                            categorical_mappings[column].keys()))))

            # Ensure Correct Data Types:
            for column in data_set.columns:
                if data_set[column].dtype == pl.Int64:
                    data_set = data_set.with_columns(
                        pl.col(column).cast(pl.Int32).alias(column))
                elif data_set[column].dtype == pl.Float64:
                    data_set = data_set.with_columns(
                        pl.col(column).cast(pl.Float32).alias(column))
                else:
                    data_set = data_set.with_columns(
                        pl.col(column).cast(pl.Utf8).alias(column))

            # Drop Unnecessary Columns
            data_set = data_set.drop(columns_to_drop_entirely)

            if (key == "train"):
                self.train_data = data_set
            elif (key == "test"):
                self.test_data = data_set

    def prepare_features(self,
                         target_column: str = 'SalePrice',
                         selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by separating features and the target variable, 
        and preprocessing them for training and testing.

        Args:
            target_column (str): Name of the target variable column. Default is 'SalePrice'.
            selected_predictors (List[str]): Specific columns to use as predictors. 
                                            If None, use all columns except the target.

        Tasks:
        1. Separate Features and Target:
            - Split the dataset into predictors (`X`) and target variable (`y`).
            - Use `selected_predictors` if provided; otherwise, use all columns except the target.
        2. Split Numeric and Categorical Features:
            - Identify numeric and categorical columns.
        3. Create a Preprocessing Pipeline:
            - Numeric Data: Impute missing values with the mean and standard scale the features.
            - Categorical Data: Impute missing values with a new category and apply one-hot encoding.
            - Use `ColumnTransformer` to combine both pipelines.
        4. Split Data:
            - Split the data into training and testing sets using `train_test_split`.

        Returns:
            - X_train, X_test, y_train, y_test: Training and testing sets.
        """

        # Separate Features and Target:
        if selected_predictors is None:
            selected_predictors = [
                col for col in self.train_data.columns if col != target_column
            ]
        X = self.train_data.select(selected_predictors).to_pandas()
        y = self.train_data.select(target_column).to_series()

        # Split Numeric and Categorical Features:
        numeric_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        # Create a Preprocessing Pipeline:
        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(
                strategy='mean')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[('imputer',
                    SimpleImputer(strategy='constant', fill_value='missing')
                    ), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        self.preprocessor = ColumnTransformer(
            transformers=[('numeric', numeric_transformer, numeric_features),
                          ('categorical', categorical_transformer,
                           categorical_features)])
        # Split Data:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
    

        return X_train, X_test, y_train, y_test

    def train_baseline_models(self) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate baseline machine learning models for house price prediction.

        Models:
        1. Linear Regression
        2. Choose One Advanced Model:
            - RandomForestRegressor
            - GradientBoostingRegressor

        Tasks:
        1. Create a Pipeline for Each Model:
            - Combine preprocessing and the estimator into a single pipeline.
        2. Train Models:
            - Train each model on the training set.
        3. Evaluate Models:
            - Use metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R²), 
            and Mean Absolute Percentage Error (MAPE).
            - Compute metrics on both training and test sets for comparison.
        4. Summarize Results:
            - Return a dictionary of model names and their evaluation metrics and the model itself.

        Returns:
            A dictionary structured like:
                {
                    "Linear Regression": 
                        { 
                            "metrics": {"MSE": ..., "R2": ..., "MAE": ..., "MAPE": ...},
                            "model": (model object)
                        },
                    "Advanced Model":
                        { 
                            "metrics": {"MSE": ..., "R2": ..., "MAE": ..., "MAPE": ...},
                            "model": (model object)
                        }
                }
        """
        X_train, X_test, y_train, y_test = self.prepare_features()
        models = {
            "Linear Regression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42)
        }
        results = {}
        for model_name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor',
                                        self.preprocessor), ('regressor',
                                                             model)])
            pipeline.fit(X_train, y_train)
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            metrics = {
                'MSE': {
                    "train": mean_squared_error(y_train, y_pred_train),
                    "test": mean_squared_error(y_test, y_pred_test)
                },
                'R2': {
                    "train": r2_score(y_train, y_pred_train),
                    "test": r2_score(y_test, y_pred_test)
                },
                'MAE': {
                    "train": mean_absolute_error(y_train, y_pred_train),
                    "test": mean_absolute_error(y_test, y_pred_test)
                },
                'MAPE': {
                    "train":
                    mean_absolute_percentage_error(y_train, y_pred_train),
                    "test":
                    mean_absolute_percentage_error(y_test, y_pred_test)
                }
            }
            results[model_name] = {"metrics": metrics, "model": pipeline}
        return results

    def forecast_sales_price(self, model_type: str = 'LinearRegression'):
        """
        Use the trained model to forecast house prices on the test dataset.

        Args:
            model_type (str): Type of model to use for forecasting. Default is 'LinearRegression'. Other option is 'Advanced'.

        Tasks:
            1. Select the Desired Model:
                - Ensure the model type is trained and available.
            2. Generate Predictions:
                - Use the selected model to predict house prices for the test set.
            3. Create a Submission File:
                - Save predictions in the required format:
                    - A CSV with columns: "Id" (from test data) and "SalePrice" (predictions).
                - Example:

                    Id,SalePrice
                    1461,200000
                    1462,175000

            4. Save the File:
                - Name the file `submission.csv` and save it in the `src/real_estate_toolkit/ml_models/outputs/` folder.

        Tips:
            - Ensure preprocessing steps are applied to the test data before making predictions.
        """
        # Select the Desired Model:
        try:
            results = self.train_baseline_models()  # Consider storing this during training
            if model_type not in results:
                raise ValueError(f"Model type '{model_type}' not found. Available models: {list(results.keys())}")
            model = results[model_type]["model"]
        except NotFittedError:
            raise RuntimeError("The specified model has not been trained. Train models before forecasting.")
        # Generate Predictions:
        try:
            X_test = self.test_data.select([
                col for col in self.test_data.columns if col != 'SalePrice'
            ]).to_pandas()
            y_pred = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess or predict: {str(e)}")
        # Create a Submission File:
        if 'Id' not in self.test_data.columns:
            raise ValueError("Test data must contain an 'Id' column for submission.")
        submission_data = {'Id': self.test_data['Id'].to_pandas(), 'SalePrice': y_pred}
        submission_df = pl.DataFrame(submission_data)

        # Save the File:
        save_file_path = "real_estate_toolkit/ml_models/outputs/submission.csv"
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        submission_df.write_csv(save_file_path, separator=",")
        print(f"Submission file saved to: {save_file_path}")
        print("Submission file preview:")
        print(submission_df)
