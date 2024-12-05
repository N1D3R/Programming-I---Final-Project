from typing import List, Dict
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import os
import pandas as pd

class MarketAnalyzer:
    def __init__(self, data_path: str):
        try:
            # Load the dataset with NA error prevention
            self.real_state_data = pl.read_csv(
                data_path,
                null_values=["NA"],  # Treat 'NA' as missing values
                infer_schema_length=10000  # Increase schema inference length for large files
            )
            self.real_state_clean_data = self.real_state_data.clone()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.real_state_clean_data = None

        # Check for `pyarrow` dependency
        try:
            import pyarrow
        except ImportError:
            print("Warning: 'pyarrow' is not installed. Some features may not work. Install it using 'pip install pyarrow'.")

    def clean_data(self) -> None:
        if self.real_state_clean_data is not None:
            try:
                for column in self.real_state_clean_data.columns:
                    # Handle numeric columns
                    if self.real_state_clean_data[column].dtype in [pl.Float64, pl.Int64]:
                        self.real_state_clean_data = self.real_state_clean_data.with_columns(
                            self.real_state_clean_data[column].fill_null(self.real_state_clean_data[column].mean())
                        )
                    # Handle string columns
                    elif self.real_state_clean_data[column].dtype == pl.Utf8:
                        self.real_state_clean_data = self.real_state_clean_data.with_columns(
                            self.real_state_clean_data[column].fill_null("Missing")
                        )

                # Cast numeric columns to float
                numeric_columns = self.real_state_clean_data.select(pl.col(pl.Float64, pl.Int64)).columns
                self.real_state_clean_data = self.real_state_clean_data.with_columns(
                    [self.real_state_clean_data[col].cast(pl.Float64) for col in numeric_columns]
                )

                # Cast categorical columns to categorical type
                categorical_columns = self.real_state_clean_data.select(pl.col(pl.Utf8)).columns
                self.real_state_clean_data = self.real_state_clean_data.with_columns(
                    [self.real_state_clean_data[col].cast(pl.Categorical) for col in categorical_columns]
                )
            except Exception as e:
                print(f"Error during data cleaning: {e}")

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        if self.real_state_clean_data is None:
            raise RuntimeError("Data has not been loaded properly.")

        try:
            # Analyze Sale Prices
            sale_prices = self.real_state_clean_data["SalePrice"]
            price_statistics = pl.DataFrame({
                "Metric": ["Mean", "Median", "Standard Deviation", "Minimum", "Maximum"],
                "Value": [
                    sale_prices.mean(),
                    sale_prices.median(),
                    sale_prices.std(),
                    sale_prices.min(),
                    sale_prices.max()
                ]
            })

            # Create histogram plot
            fig = px.histogram(
                self.real_state_clean_data.to_pandas(),
                x="SalePrice",
                title="Distribution of Sale Prices",
                labels={"SalePrice": "Sale Price"},
                template="plotly_white"
            )
            fig.show()  # Show the plot directly in the application
            return price_statistics
        except ImportError as e:
            print(f"Price distribution analysis failed: {e}. Ensure required modules like 'pyarrow' are installed.")
            return pl.DataFrame()
        except Exception as e:
            print(f"An error occurred during price distribution analysis: {e}")
            return pl.DataFrame()

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        if self.real_state_clean_data is None:
            raise RuntimeError("Data has not been loaded properly.")

        try:
            neighborhood_stats = []
            for neighborhood in self.real_state_clean_data["Neighborhood"].unique():
                filtered_data = self.real_state_clean_data.filter(pl.col("Neighborhood") == neighborhood)
                stats = {
                    "Neighborhood": neighborhood,
                    "Median Price": filtered_data["SalePrice"].median(),
                    "Average Price": filtered_data["SalePrice"].mean(),
                    "Price Spread": filtered_data["SalePrice"].std(),
                    "Minimum Price": filtered_data["SalePrice"].min(),
                    "Maximum Price": filtered_data["SalePrice"].max(),
                }
                neighborhood_stats.append(stats)
            neighborhood_stats = pl.DataFrame(neighborhood_stats)

            # Create boxplot
            fig = px.box(
                self.real_state_clean_data.to_pandas(),
                x="Neighborhood",
                y="SalePrice",
                title="Sale Price Distribution by Neighborhood",
                labels={"SalePrice": "Sale Price", "Neighborhood": "Neighborhood"},
                template="plotly_white"
            )
            fig.update_layout(xaxis={"categoryorder": "total ascending"})
            fig.show()  # Show the plot directly in the application

            return neighborhood_stats
        except Exception as e:
            print(f"An error occurred during neighborhood price comparison: {e}")
            return pl.DataFrame()

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        if self.real_state_clean_data is None:
            raise RuntimeError("Data has not been loaded properly.")

        try:
            if not all(var in self.real_state_clean_data.columns for var in variables):
                raise ValueError("One or more variables do not exist in the dataset.")

            selected_data = self.real_state_clean_data.select(variables).to_pandas()
            correlation_matrix = selected_data.corr()

            import plotly.figure_factory as ff
            fig = ff.create_annotated_heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns.tolist(),
                y=correlation_matrix.index.tolist(),
                colorscale="Viridis",
                showscale=True
            )
            fig.update_layout(title="Feature Correlation Heatmap")
            fig.show()  # Show the heatmap directly in the application
        except Exception as e:
            print(f"An error occurred during heatmap generation: {e}")

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        figures = {}

        if self.real_state_clean_data is None:
            raise RuntimeError("Data has not been loaded properly.")

        try:
            # Scatter plots
            fig1 = px.scatter(
                self.real_state_clean_data.to_pandas(),
                x="GrLivArea",
                y="SalePrice",
                color="Neighborhood",
                title="House Price vs. Total Square Footage",
                labels={"GrLivArea": "Total Square Footage", "SalePrice": "Sale Price"},
                trendline="ols"
            )
            fig1.show()  # Show the scatter plot directly in the application
            figures["House Price vs. Total Square Footage"] = fig1

            fig2 = px.scatter(
                self.real_state_clean_data.to_pandas(),
                x="YearBuilt",
                y="SalePrice",
                color="Neighborhood",
                title="Sale Price vs. Year Built",
                labels={"YearBuilt": "Year Built", "SalePrice": "Sale Price"},
                trendline="ols"
            )
            fig2.show()  # Show the scatter plot directly in the application
            figures["Sale Price vs. Year Built"] = fig2

            fig3 = px.scatter(
                self.real_state_clean_data.to_pandas(),
                x="OverallQual",
                y="SalePrice",
                color="Neighborhood",
                title="Overall Quality vs. Sale Price",
                labels={"OverallQual": "Overall Quality", "SalePrice": "Sale Price"},
                trendline="ols"
            )
            fig3.show()  # Show the scatter plot directly in the application
            figures["Overall Quality vs. Sale Price"] = fig3

            return figures
        except Exception as e:
            print(f"An error occurred during scatter plot generation: {e}")
            return {}
