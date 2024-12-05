from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint
from typing import Optional, List, Dict, Union, Any
from dataclasses import dataclass
from pathlib import Path
import random
from agent_based_model.houses import House
from agent_based_model.house_market import HousingMarket
from agent_based_model.consumers import Segment, Consumer


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
    order: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def __init__(self,
                 housing_market_data: List[Dict[str, Any]],
                 consumers_number: int,
                 years: int,
                 annual_income: AnnualIncomeStatistics,
                 children_range: ChildrenRange,
                 order: CleaningMarketMechanism,
                 down_payment_percentage: float = 0.2,
                 saving_rate: float = 0.3,
                 interest_rate: float = 0.05):
        for var, var_type in [(consumers_number, int), (years, int),
                              (annual_income, AnnualIncomeStatistics),
                              (children_range, ChildrenRange),
                              (order, CleaningMarketMechanism),
                              (down_payment_percentage, float),
                              (saving_rate, float), (interest_rate, float)]:
            
            if not isinstance(var, var_type):
                raise TypeError(f"Unexpected variable type in '{var}'")

        self.housing_market_data = housing_market_data
        self.consumers_number = consumers_number
        self.years = years
        self.annual_income = annual_income
        self.children_range = children_range
        self.order = order

    def create_housing_market(self):
      """
      Initialize market with houses.

      Implementation tips:
      - Convert raw data to House objects
      - Validate input data
      - Use a for loop for create the List[Consumers] object needed to use the housing market.
      - Assign self.housing_market to the class.
      """

      houses = []
      consumers = []

      # Validate house data
      for house_data in self.housing_market_data:
          required_keys = ["id", "sale_price", "gr_liv_area", "garage_area", 
                          "wood_deck_sf", "open_porch_sf", "enclosed_porch", 
                          "3ssn_porch", "screen_porch", "pool_area", 
                          "total_bsmt_sf", "bedroom_abv_gr", "year_built"]
          if not all(key in house_data for key in required_keys):
              raise ValueError(f"Missing keys in house data: {house_data}")
              
          if not isinstance(house_data["sale_price"], (int, float)):
              raise TypeError(f"Invalid type for sale_price: {house_data['sale_price']}")

          house_area = (int(house_data["gr_liv_area"]) + int(house_data["garage_area"]) +
                        int(house_data["wood_deck_sf"]) +
                        int(house_data["open_porch_sf"]) +
                        int(house_data["enclosed_porch"]) +
                        int(house_data["3ssn_porch"]) + int(house_data["screen_porch"]) +
                        int(house_data["pool_area"]) + int(house_data["total_bsmt_sf"]))

          house = House(
              house_data["id"],
              house_data["sale_price"],
              house_area,
              house_data["bedroom_abv_gr"],
              house_data["year_built"],
              quality_score=None
          )

          houses.append(house)

      # Create consumers
      """for consumer_data in self.consumers_data:
          required_consumer_keys = ["id", "annual_income", "children_number", "segment"]
          if not all(key in consumer_data for key in required_consumer_keys):
              raise ValueError(f"Missing keys in consumer data: {consumer_data}")

          if not isinstance(consumer_data["annual_income"], (int, float)):
            raise TypeError(f"Invalid type for annual_income: {consumer_data['annual_income']}")


          consumer = Consumer(
              consumer_data["id"],
              consumer_data["annual_income"],
              consumer_data["children_number"],
              consumer_data["segment"],
          )
          consumers.append(consumer)

      
      self.consumers = consumers """
      self.housing_market = HousingMarket(houses)
        
    def create_consumers(self) -> None:
        """
      Generate consumer population.

      Implementation instructions:
      1. Create a list of consumers with length equal to consumers_number. Make it a property of the class simulation (self.consumers = ...).
      2. Assign a randomly generated annual income using a normal distribution (gauss from random)
         truncated by maximum and minimum (while value > maximum or value < min sample again) considering AnnualIncomeStatistics
      3. Assign a randomly generated children number using a random integer generator (randint from random)
      4. Assign randomly a segment (Segment value from consumer). All segments have the same probability of being assigned.
      5. Assign the simulation saving rate (simulated economy saving rate).

      Implementation tips:
      - Use provided statistical distributions
      - Ensure realistic attribute values
      - Assign segments appropriately
      """

        self.consumers = []
        segments = list(Segment)
        
        for id in range(self.consumers_number):

            annual_income = random.gauss(self.annual_income.average,
                                         self.annual_income.standard_deviation)
            while annual_income > self.annual_income.maximum or annual_income < self.annual_income.minimum:
                annual_income = random.gauss(
                    self.annual_income.average,
                    self.annual_income.standard_deviation)

            children_number = random.randint(self.children_range.minimum,
                                             self.children_range.maximum)

            segment = random.choice(segments)

            consumer = Consumer(id, annual_income, children_number, segment)
            consumer.saving_rate = self.saving_rate
            self.consumers.append(consumer)

    def compute_consumers_savings(self) -> None:
        """
      Calculate savings for all consumers.

      Implementation tips:
      - Apply saving rate consistently to all consumers.
      - Handle edge cases
      """
        if not self.consumers:
            print("No consumers to process.")
            return

        if not isinstance(self.years, int) or self.years <= 0:
            raise ValueError("Invalid value for years: must be a positive integer.")

        for consumer in self.consumers:
            consumer.compute_savings(self.years)

    def sortingCriteria(self, consumer):
        return consumer.annual_income

    def clean_the_market(self) -> None:
        """
      Execute market transactions.

      Implementation tips:
      - Use buy_a_house function for each consumer
      - Implement ordering mechanisms (CleaningMarketMechanism)
      - Track successful purchases
      - Handle market clearing

      Docs checked:
      - https://www.w3schools.com/python/ref_list_sort.asp
      """

        if self.order == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=sortingCriteria, reverse=True)
        elif self.order == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=sortingCriteria)
        elif self.order == CleaningMarketMechanism.RANDOM:
            random.shuffle(self.consumers)
        else:
            raise ValueError("Invalid cleaning market mechanism provided!")

        successful_purchases = 0
        
        for consumer in self.consumers:
            house = consumer.buy_a_house(self.housing_market)
            if house:
                successful_purchases += 1
                
        print(f"Market cleaning completed. Successful purchases: {successful_purchases}")


    def compute_owners_population_rate(self) -> float:
        """
      Compute the owners population rate after the market is clean.

      Implementation tips:
      - Total consumers who bought a house over total consumers number
      """

        owners = 0
        for consumer in self.consumers:
            if consumer.house is not None:
                owners += 1

        if len(self.consumers) > 0:
            return owners / len(self.consumers)
        else:
            return 0

    def compute_houses_availability_rate(self) -> float:
        """
      Compute the houses availability rate after the market is clean.

      Implementation tips:
      - Houses available over total houses number
      """

        available_houses = 0
        for house_id, house in self.housing_market.houses.items():
            if house.available:
                available_houses += 1

        total_houses = len(self.housing_market.houses)

        if total_houses > 0:
            return available_houses / total_houses
        else:
            return 0
