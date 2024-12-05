from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict
from agent_based_model.houses import House, QualityScore

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

    def __init__(
        self,
        id: int,
        annual_income: float,
        children_number: int,
        segment: Segment,
        house: Optional[House] = None,
        savings: float = 0.0,
        saving_rate: float = 0.3,
        interest_rate: float = 0.05
      ):

      for var in [
          id,
          annual_income,
          children_number,
          savings,
          saving_rate,
          interest_rate
      ]:
        if not isinstance(var, (float,int)):
            raise TypeError(f"Consumer {var} must be an integer!")

      self.id = id
      self.annual_income = annual_income
      self.children_number = children_number
      self.segment = segment
      self.house = house

    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time.

        Implementation tips:
        - Use compound interest formula
        - Consider annual calculations
        - Account for saving_rate
        """
        current_savings = self.savings

        for year in range(years):
            current_savings *= (1 + self.interest_rate)
            current_savings += self.annual_income * self.saving_rate

        self.savings = current_savings

    def buy_a_house(self, housing_market) -> None:
        """
        Attempt to purchase a suitable house.

        Implementation tips:
        - Check savings against house prices
        - Consider down payment requirements
        - Match house to family size needs
        - Apply segment-specific preferences
        """

        suitable_houses = []

        for house in housing_market.houses.values():
            if not house.available:
                continue

            if self.segment == Segment.FANCY:
                if house.is_new_construction() and house.quality_score == QualityScore.EXCELLENT:
                    suitable_houses.append(house)
            elif self.segment == Segment.OPTIMIZER:
                price_per_sqft = house.calculate_price_per_square_foot()
                if price_per_sqft < (self.annual_income / 12):
                    suitable_houses.append(house)
            elif self.segment == Segment.AVERAGE:
                if house.price <= housing_market.calculate_average_price():
                    suitable_houses.append(house)
                    
        if suitable_houses:
            for house in suitable_houses:
                down_payment = 0.2 * house.price
                
                if self.savings >= down_payment and house.bedrooms >= self.children_number + 1: #childen's and parents rooms
                    self.house = house
                    self.savings -= down_payment
                    house.sell_house()
                    print(f"Consumer {self.id} bought the house with ID {house.id}.")
                    return house
            

        else:
            print(f"Consumer {self.id} could not find a suitable house to buy.")
