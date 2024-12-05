from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime
import warnings

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

    def _init__(
        self,
        id: int,
        price: float,
        area: float,
        bedrooms: int,
        year_built: int,
        quality_score: Optional[QualityScore] = None,
        available: bool = True
    ):

        for var in [
            id,
            price,
            area,
            bedrooms,
            year_built
        ]:
          if not isinstance(var, (float,int)):
              raise TypeError(f"House {var} must be an integer!")

        self.id = id
        self.price = price
        self.area = area
        self.bedrooms = bedrooms
        self.year_built = year_built
        self.quality_score = quality_score
        self.available = available

    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.

        Implementation tips:
        - Divide price by area
        - Round to 2 decimal places
        - Handle edge cases (e.g., area = 0)
        """

        if self.area == 0:
            raise ValueError("House area cannot be zero!")

        square_foot_price = round(self.price / self.area, 2)
        return square_foot_price

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if house is considered new construction (< 5 years old).

        Implementation tips:
        - Compare current_year with year_built
        - Consider edge cases for very old houses
        """

        if self.year_built > current_year:
            raise ValueError("House year built cannot be in the future!")

        if not isinstance(current_year, (int, float)):
            raise TypeError("Current year must be a number!")

        return (current_year - self.year_built) < 5

    def get_quality_score(self) -> None:
        """
        Generate a quality score based on house attributes.

        Implementation tips:
        - Consider multiple factors (age, size, bedrooms)
        - Create meaningful score categories
        - Handle missing quality_score values

        How to calculate quality score:
        - Age: Newer houses have higher quality
        - Size: Larger houses have higher quality
        - Bedrooms: Houses with more bedrooms have higher quality
        """

        if self.quality_score is not None:
            warnings.warn("Quality score already assigned, overwriting existing value.")

        current_year = datetime.now().year
        house_age = current_year - self.year_built


        if house_age < 5:
            house_age_factor = 5
        elif house_age < 20:
            house_age_factor = 4
        elif house_age < 50:
            house_age_factor = 3
        else:
            house_age_factor = 2

        if self.area > 2500:
            size_factor = 5
        elif self.area > 1500:
            size_factor = 4
        else:
            size_factor = 3

        if self.bedrooms >= 3:
            bedroom_factor = 4
        else:
            bedroom_factor = 2

        total_score = house_age_factor + size_factor + bedroom_factor

        if total_score >= 12:
            self.quality_score = QualityScore.EXCELLENT
        elif total_score >= 10:
            self.quality_score = QualityScore.GOOD
        elif total_score >= 8:
            self.quality_score = QualityScore.AVERAGE
        elif total_score >= 6:
            self.quality_score = QualityScore.FAIR
        else:
            self.quality_score = QualityScore.POOR


    def sell_house(self) -> None:
        """
        Mark house as sold.

        Implementation tips:
        - Update available status
        """
        self.available = False


    # In house_market.py or wherever you process the data
