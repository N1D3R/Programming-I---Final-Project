# Real Estate Analysis Toolkit Project

## 1. Overview

Students will create a housing market analysis toolkit that combines fundamental Python concepts with agent-based modelling and modern data analysis libraries. The project will be structured as a Python package (sort of) named `real_estate_toolkit`.

The project will use the Ames Housing dataset. This dataset covers a wealth of features that are critical for analyzing house prices, with numerous categories capturing structure types, neighborhood characteristics, and utility details.

### 1.1 Learning Objectives
- Apply fundamental Python concepts in a real-world scenario
- Create a well-structured Python code
- Implement a basic Agent-Based model for learning about the real-estate market
- Implement data analysis workflows
- Practice clean code principles and type hints
- Utilize modern data analysis libraries
- Implement machine learning pipelines

### 1.2 Delivery date:
Students must submit their final project by Sunday, December 15, 2024, before 23:59:59.

### 1.3 Delivery format:
One designated member of the group should email a link to a Google Drive folder containing the completed project code to josefernando.moreno@upf.edu. Ensure that the folder permissions allow for viewing and downloading. Please include the names of all group members in the email for easy identification.

For an **additional bonus of +1 point** on the final grade, you can also create a GitHub repository for the project. This repository should include the complete code. To qualify for the bonus:

- Ensure that all code and project assets are uploaded and well-organized within the repository.
- Use Git for version control, with meaningful commit messages that document key stages of your project’s development.
- Instead of the Google Drive link share the link to the GitHub repository in the email.

*Taking this extra step will not only enhance the structure and presentation of your project but also give you practical experience with Git and GitHub, valuable tools in collaborative development.*

## 2. The dataset

The dataset was taken from the [Kaggle](https://www.kaggle.com/) competition ["House Prices - Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) which challenges participants to predict residential housing prices based on a variety of features. The dataset includes information on 79 explanatory variables, which capture various aspects of residential homes, such as the number of rooms, square footage, neighborhood, and quality of materials used. 

### 2.1 The data

The complete [dataset for the "House Prices - Advanced Regression Techniques" competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) on Kaggle contains two main CSV files:

- **data_description.txt** - This file provides detailed explanations for each feature in the dataset, helping to understand and interpret the variables. Read it carefully.

- **train.csv** - This file has 1,460 rows (houses) and 81 columns. It includes 79 feature columns describing various aspects of each house, a unique identifier column (Id), and the SalePrice column, which is the target variable representing the sale price of each home in USD.

- **test.csv** - This file has 1,459 rows and the same 80 columns as the training data, except it lacks the SalePrice column. Participants use this file to generate their predictions.


#### Key Features
The features cover a wide range of information about each home, such as:

- **Location & Size:** Lot area, neighborhood, and lot frontage.
- **House Details:** Year built, type of dwelling, number of stories, and basement.
- **Rooms & Utilities:** Number of bathrooms, bedrooms, kitchen quality, fireplaces, and air conditioning.
- **Material & Quality:** Condition, quality of finishes, and type of materials used for roofing, siding, and floors.
- **Outdoor & Surroundings:** Garage, porch, and fencing details.
- **Price:** The target metric.


### 2.2 The competition

**You don't have to participate in the competition**, but I encourage you to enrol and submit your results. [Kaggle](https://www.kaggle.com/) is a fantastic platform for building your career portfolio and learning about data science, analytics, and, of course, Python. Furthermore, along the project, you will develop a model that you can use to submit your result. Later in the project, you will see that submitting your result can increase your project grade.

Participants of the competition are tasked with building a model that accurately estimates house sale prices. This competition is ideal for exploring regression techniques, feature engineering, and model evaluation. It's geared towards beginners and experienced data scientists, as it involves a blend of data preprocessing, handling missing values, and advanced machine learning algorithms like linear regression, decision trees, and ensemble methods. The goal is to minimize the prediction error measured by Root Mean Squared Logarithmic Error (RMSLE). This project offers a practical learning opportunity in predictive modelling


## 3. The project

### 3.1 Part 1 (50%): House Market Explorer and Modeler - Project Setup, Apply Basic Python Concepts and Generate your First Agent-Based Model

#### Initial Project Structure
```
real_estate_toolkit/
├── pyproject.toml
├── README.md
├── .venv/
└── src/
    └── real_estate_toolkit/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   ├── loader.py
        │   ├── cleaner.py
        │   └── descriptor.py
        ├── agent_based_model/
        │   ├── __init__.py
        │   ├── consumers.py
        │   ├── houses.py
        │   ├── house_market.py
        │   └── simulation.py
        └── main.py
```

**In this part of the project you can't use Polars.**

#### 3.1.1: Project Initialization (5%)
Use poetry to initialize your project. Remember that if you need or are planning to use other libraries, you need to add them. Here is a basic setup example:

```bash
poetry config virtualenvs.in-project true
poetry new real_estate_toolkit --src
cd real_estate_toolkit
poetry add numpy polars plotly scikit-learn
poetry add pytest --group dev
poetry install
```

With these commands, you should get a `pyproject.toml` file like this. Add your group member names and emails.

```toml
[tool.poetry]
name = "real_estate_toolkit"
version = "0.1.0"
description = "A toolkit for real estate data analysis"
authors = ["Student Name <student@email.com>"]

[tool.poetry.dependencies]
...

[tool.poetry.group.dev.dependencies]
..
```

Remember that you can use your virtual env with the following command:

```bash
poetry shell
```

#### 3.1.2: Create Data Base Classes and Functions (20%)

##### 3.1.2.1 Implement the foundation classes that will be used throughout the explorer (15%):

Implement the following files and their classes and the methods proposed applying your knowledge about classes, lists, dictionaries and file I/O. Use best practices about variable, function and class naming. Use type hints.

- **`src/real_estate_toolkit/data/loader.py`:**

    ```python
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Dict, List, Union

    @dataclass
    class DataLoader:
        """Class for loading and basic processing of real estate data."""
        data_path: Path
        
        def load_data_from_csv(self) -> List[Dict[str, Any]]:
            """Load data from CSV file into a list of dictionaries."""
            return ... # List of dictionaries with the data
        
        def validate_columns(self, required_columns: List[str]) -> bool:
            """Validate that all required columns are present in the dataset."""
            return ... # True if all required columns are present
    ```
    For this file, students are provided with a partially implemented `DataLoader` class, designed to load and validate real estate data from a CSV file. The class has two main methods that need to be fully implemented.

    - `load_data_from_csv`: This method is intended to read data from a CSV file located at data_path and return the data as a list of dictionaries, where each dictionary represents a row, mapping column names to values. Students should use Polars to handle this CSV loading efficiently and ensure the data is structured as expected.

    - `validate_columns`: This method should confirm that the dataset contains all the columns specified in required_columns. It should return True only if all required columns are present in the dataset, otherwise False. Students should implement logic to check for each required column's presence, ensuring that the data integrity is maintained.

- **`src/real_estate_toolkit/data/cleaner.py`:**
    ```python
    from dataclasses import dataclass
    from typing import Dict, List

    @dataclass
    class Cleaner:
        """Class for cleaning real estate data."""
        data: List[Dict[str, Any]]
        
        def rename_with_best_practices(self) -> None:
            """Rename the columns with best practices (e.g. snake_case very descriptive name)."""
        
        def na_to_none(self) -> List[Dict[str, Any]]:
            """Replace NA to None in all values with NA in the dictionary."""
    ```
    In this exercise, students are provided with a partially implemented Cleaner class, designed for cleaning real estate data. This class includes two methods that need further implementation:

    - `rename_with_best_practices`: This method should rename each column in the dataset following best practices, specifically using snake_case and ensuring column names are descriptive. Students should implement logic to iterate through the column names in data and apply transformations to make names clear, consistent, and in snake_case.

    - `na_to_none`: This method should replace any occurrence of the string "NA" with Python's None in all the dataset’s entries. The method should return a modified list of dictionaries where all occurrences of "NA" in any column are replaced by None, making the data easier to handle for later stages of analysis.

- **`src/real_estate_toolkit/data/descriptor.py`:**

    ```python
    from dataclasses import dataclass
    from typing import Dict, List, Tuple

    @dataclass
    class Descriptor:
        """Class for cleaning real estate data."""
        data: List[Dict[str, Any]]

        def none_ratio(self, columns: List[str] = "all"):
            """Compute the ratio of None value per column.
            If columns = "all" then compute for all.
            Validate that column names are correct. If not make an exception.
            Return a dictionary with the key as the variable name and value as the ratio.
            """
        
        def average(self, columns: List[str] = "all") -> Dict[str, float]:
            """Compute the average value for numeric variables. Omit None values.
            If columns = "all" then compute for all numeric ones.
            Validate that column names are correct and correspond to a numeric variable. If not make an exception.
            Return a dictionary with the key as the numeric variable name and value as the average
            """
        
        def median(self, columns: List[str] = "all") -> Dict[str, float]:
            """Compute the median value for numeric variables. Omit None values.
            If columns = "all" then compute for all numeric ones.
            Validate that column names are correct and correspond to a numeric variable. If not make an exception.
            Return a dictionary with the key as the numeric variable name and value as the average
            """
        
        def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
            """Compute the percentile value for numeric variables. Omit None values.
            If columns = "all" then compute for all numeric ones.
            Validate that column names are correct and correspond to a numeric variable. If not make an exception.
            Return a dictionary with the key as the numeric variable name and value as the average
            """
        
        def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
            """Compute the mode for variables. Omit None values.
            If columns = "all" then compute for all.
            Validate that column names are correct. If not make an exception.
            Return a dictionary with the key as the variable name and value as a tuple of the variable type and the mode.
            If the variable is categorical
            """
    ```

    In this exercise, students are provided with a partially implemented `Descriptor` class, which focuses on summarizing and describing real estate data. The class includes multiple methods that need to be completed:

    - `none_ratio`: This method calculates the ratio of `None` values in each specified column. If `columns="all"`, it should compute this for all columns. Students should validate that each specified column name is correct, raising an exception for any invalid names. The method should return a dictionary where each key is a column name and each value is the ratio of `None` values in that column.

    - `average`: This method computes the average value for numeric variables in the specified columns, ignoring `None` values. If `columns="all"`, it should calculate the average for all numeric columns. Students should ensure each column is a valid numeric variable, raising an exception if not, and return a dictionary mapping each numeric variable name to its average value.

    - `median`: Similar to average, this method calculates the median for numeric columns, ignoring `None` values. Students should validate that the specified columns are numeric and raise an exception for invalid names or types. The method should return a dictionary with each numeric variable's median.

    - `percentile`: This method computes a specific percentile for numeric columns, based on the percentile parameter, which defaults to 50 (i.e., the median). If `columns="all"`, the method should compute for all numeric columns. Students need to validate column names and types, returning a dictionary where keys are column names and values are the computed percentile values.

    - `type_and_mode`: This method finds the mode (most frequent value) for each specified column, excluding `None` values. If `columns="all"`, it should compute for all columns. The method should validate column names and determine the variable type, returning a dictionary where each key is a column name and each value is a tuple with the type of the variable and its mode. For categorical variables, the tuple should contain the variable type and mode as a string.

##### 3.1.2.2 Implement descriptor class using NumPy (5%):

Generate another class called `DescriptorNumpy` and use *NumPy* instead of basic data structures for all the functions in `Descriptor` class. The functions have to return and use *NumPy* types.

#### 3.1.3: Create Agent-Based Model Classes and Functions (25%)

This part of the project involves creating an agent-based model (ABM) to simulate a real estate market. You will implement various classes representing houses, market dynamics, and consumer behavior to understand how different factors influence housing market outcomes. By completing this piece, you will:

- Apply object-oriented programming principles in Python
- Implement an agent-based model with multiple interacting components
- Use Python's type hinting system for better code documentation
- Model real-world market dynamics and decision-making processes
- Practice working with random distributions and synthetic data generation

##### 3.1.3.1: Create a House Class
In `house.py`, implement a `House` class. The `House` class represents individual properties in the market. The required properties are:

- `id`: Unique identifier for each house
- `price`: Sale price of the house
- `area`: Square footage of the house
- `bedrooms`: Number of bedrooms
- `year_built`: Year of construction
- `quality_score`: Optional rating of house quality
- `available`: Boolean indicating if house is available for sale

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict

class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1

@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore]
    available: bool = True
    
    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.
        
        Implementation tips:
        - Divide price by area
        - Round to 2 decimal places
        - Handle edge cases (e.g., area = 0)
        """
    
    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if house is considered new construction (< 5 years old).
        
        Implementation tips:
        - Compare current_year with year_built
        - Consider edge cases for very old houses
        """
    
    def get_quality_score(self) -> None:
        """
        Generate a quality score based on house attributes.
        
        Implementation tips:
        - Consider multiple factors (age, size, bedrooms)
        - Create meaningful score categories
        - Handle missing quality_score values
        """

    def sell_house(self) -> None:
        """
        Mark house as sold.
        
        Implementation tips:
        - Update available status 
        """
```

##### 3.1.3.2: Create Market Collection
In `market.py`, implement a `HousingMarket` class. The `HousingMarket` class manages collections of houses and market-wide operations.

```python
from typing import List, Dict, Optional
from .house import House

class HousingMarket:
    def __init__(self, houses: List[House]):
        self.houses: List[House] = houses
    
    def get_house_by_id(self, house_id: int) -> House:
        """
        Retrieve specific house by ID.
        
        Implementation tips:
        - Use efficient search method
        - Handle non-existent IDs
        """
    
    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate average house price, optionally filtered by bedrooms.
        
        Implementation tips:
        - Handle empty lists
        - Consider using statistics module
        - Implement bedroom filtering efficiently
        """
    
    def get_houses_that_meet_requirements(self, max_price: int, segment: str) -> Optional[List[House]]:
        """
        Filter houses based on buyer requirements.
        
        Implementation tips:
        - Consider multiple filtering criteria
        - Implement efficient filtering
        - Handle case when no houses match
        """
```

##### 3.1.3.3: Create a Consumer Agent
In `consumer.py`, implement a `Consumer` class. The `Consumer` class represents potential buyers in the market. The consumer segments are:

- `FANCY`: Prefers new construction with high quality scores
- `OPTIMIZER`: Focuses on price per square foot value
- `AVERAGE`: Considers average market prices

The `saving_rate` corresponds to the amount of annual income reserved for savings.

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict
from .house import House
from .market import HousingMarket

class Segment(Enum):
    FANCY = auto() # House is new construction and house score is the highest
    OPTIMIZER = auto() # Price per square foot is less than monthly salary
    AVERAGE = auto() # House price is below the average housing market price

@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House]
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05
    
    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time.
        
        Implementation tips:
        - Use compound interest formula
        - Consider annual calculations
        - Account for saving_rate
        """

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house.
        
        Implementation tips:
        - Check savings against house prices
        - Consider down payment requirements
        - Match house to family size needs
        - Apply segment-specific preferences
        """
```

##### 3.1.3.4: Implement a Simulation Class and Functions
In `simulation.py`, implement a `Simulation` class to generate Agent-Based Model simulations. The `Simulation` class orchestrates the overall market simulation. The market cleaning mechanisms are:

- `INCOME_ORDER_DESCENDANT`: Highest income buyers first
- `INCOME_ORDER_ASCENDANT`: Lowest income buyers first
- `RANDOM`: Random order of buyers

```python
from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint
from typing import Optional, List, Dict
from .house import House
from .market import HousingMarket
from .consumer import Segment, Consumer

class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()

@dataclass
class AnnualIncomeStatistics:
    minimum: float
    average: float
    standard_deviation: float
    maximum: float

@dataclass
class ChildrenRange:
    minimum: float = 0
    maximum: float = 5

@dataclass
class Simulation:
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05
    
    def create_housing_market(self):
        """
        Initialize market with houses.
        
        Implementation tips:
        - Convert raw data to House objects
        - Validate input data
        - Assign self.housing_market to the class.
        """

    def create_consumers(self) -> None:
        """
        Generate consumer population.

        Implementation instructions:
        1. Create a list of consumers with length equal to consumers_number. Make it a property of the class simulation (self.consumers = ...).
        2. Assign a randomly generated annual income using a normal distribution (gauss from random) truncated by maximum and minimum (while value > maximum or value < min sample again) considering AnnualIncomeStatistics
        3. Assign a randomly generated children number using a random integer generator (randint from random)
        4. Assign randomly a segment (Segment value from consumer). All segments have the same probability of being assigned.
        5. Assign the simulation saving rate (simulated economy saving rate).
        
        Implementation tips:
        - Use provided statistical distributions
        - Ensure realistic attribute values
        - Assign segments appropriately
        """
    
    def compute_consumers_savings(self) -> None:
        """
        Calculate savings for all consumers.
        
        Implementation tips:
        - Apply saving rate consistently to all consumers.
        - Handle edge cases
        """

    def clean_the_market(self) -> None:
        """
        Execute market transactions.
        
        Implementation tips:
        - Use buy_a_house function for each consumer
        - Implement ordering mechanisms (CleaningMarketMechanism)
        - Track successful purchases
        - Handle market clearing
        """
    
    def compute_owners_population_rate(self) -> float:
        """
        Compute the owners population rate after the market is clean. 
        
        Implementation tips:
        - Total consumers who bought a house over total consumers number
        """
    
    def compute_houses_availability_rate(self) -> float:
        """
        Compute the houses availability rate after the market is clean. 
        
        Implementation tips:
        - Houses available over total houses number
        """
```

### 3.2 Part 2 (50%): House Market Analytics and Forecasting - Use Python for Data Analytics and Machine Learning

#### Final Project Structure
```
real_estate_toolkit/
├── pyproject.toml
├── README.md
├── .venv/
└── src/
    └── real_estate_toolkit/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   ├── loader.py
        │   ├── cleaner.py
        │   └── descriptor.py
        ├── agent_based_model/
        │   ├── __init__.py
        │   ├── consumers.py
        │   ├── houses.py
        │   ├── house_market.py
        │   └── simulation.py
        ├── analytics/
        │   ├── __init__.py
        │   ├── outputs/
        │   │   ├── (output 1 name).html
        │   │   ├── (output 2 name).html
        │   │   └── (output ... name).html
        │   └── exploratory.py
        ├── ml_models/
        │   ├── __init__.py
        │   └── predictor.py
        └── main.py
```


### 3.2.1 Data Analysis with Polars and Plotly (20%)

Create a new folder `src/real_estate_toolkit/analytics`. For graphs use an output folder `src/real_estate_toolkit/analytics/outputs/`. Use best practices for naming output files.

#### 3.2.1.1 Data Preparation and Exploratory Data Analysis

Create a new module `src/real_estate_toolkit/analytics/exploratory.py` with the following requirements:

```python
from typing import List, Dict, Any, Optional
import polars as pl
import plotly.express as px


class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        
        Args:
            data_path (str): Path to the Ames Housing dataset
        """
        self.real_state_data = pl.read_csv(...)
        self.real_state_clean_data = None
    
    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning:
        
        Tasks to implement:
        1. Identify and handle missing values
            - Decide strategy for each column (drop, fill with mean/median, no change)
        2. Convert columns to appropriate data types if needed.
            - Ensure numeric columns are numeric
            - Ensure categorical columns are categorized
        
        Returns:
            Cleaned and preprocessed dataset assigned to self.real_state_clean_data
        """
    
    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        
        Tasks to implement:
        1. Compute basic price statistics and generate another data frame called price_statistics:
            - Mean
            - Median
            - Standard deviation
            - Minimum and maximum prices
        2. Create an interactive histogram of sale prices using Plotly.
        
        Returns:
            - Statistical insights dataframe
            - Save Plotly figures for price distribution in src/real_estate_toolkit/analytics/outputs/ folder.
        """
    
    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across different neighborhoods.
        
        Tasks to implement:
        1. Group data by neighborhood
        2. Calculate price statistics for each neighborhood
        3. Create Plotly boxplot with:
            - Median prices
            - Price spread
            - Outliers
        
        Returns:
            - Return neighborhood statistics dataframe
            - Save Plotly figures for neighborhood price comparison in src/real_estate_toolkit/analytics/outputs/ folder.
        """
    
    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for variables input.
        
        Tasks to implement:
        1. Pass a list of numerical variables
        2. Compute correlation matrix and plot it
        
        Args:
            variables (Lis[str]): List of variables to correlate
        
        Returns:
            Save Plotly figures for correlation heatmap in src/real_estate_toolkit/analytics/outputs/ folder.
        """
    
    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring relationships between key features.
        
        Scatter plots to create:
        1. House price vs. Total square footage
        2. Sale price vs. Year built
        3. Overall quality vs. Sale price
        
        Tasks to implement:
        - Use Plotly Express for creating scatter plots
        - Add trend lines
        - Include hover information
        - Color-code points based on a categorical variable
        - Save them in in src/real_estate_toolkit/analytics/outputs/ folder.
        
        Returns:
            Dictionary of Plotly Figure objects for different scatter plots. 
        """
```

### 3.2.2 Machine Learning Models with Scikit-Learn (30%)

#### 3.2.2.1 Predictive Modeling Module

Create `src/real_estate_toolkit/ml_models/predictor.py`.

```python
from typing import List, Dict, Any
# Modules you can use (not all are mandatory):
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import polars as pl  # Polars should be used for data handling.


class HousePricePredictor:
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
        self.train_data = pl.read_csv(train_data_path)
        self.test_data = pl.read_csv(test_data_path)
    
    def clean_data(self):
        """
        Perform comprehensive data cleaning on the training and testing datasets.
        
        Tasks:
        1. Handle Missing Values:
            - Use a strategy for each column: drop, fill with mean/median/mode, or create a separate category.
        2. Ensure Correct Data Types:
            - Convert numeric columns to float/int.
            - Convert categorical columns to string.
        3. Drop Unnecessary Columns:
            - Identify and remove columns with too many missing values or irrelevant information.
        
        Tips:
            - Use Polars for data manipulation.
            - Implement a flexible design to allow column-specific cleaning strategies.
        """
    
    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
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
```

## 4. Evaluation via main.py

I will use the main.py file in this repo. This main.py file should run without errors as the main file in your project. If it is not the case the project will be reviewed file by file; here the grade will be "more" subjective.
