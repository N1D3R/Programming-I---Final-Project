from typing import List, Dict, Optional
from agent_based_model.consumers import Segment
from agent_based_model.houses import House, QualityScore
import statistics
from statistics import mean


class HousingMarket:
    houses: Dict[int, House]

    def __init__(self, houses: List[House]):
        house_list: Dict[int, House] = {}
        
        for house in houses:
            house_list[house.id] = house

        self.houses = house_list

    def get_house_by_id(self, house_id: int) -> Optional[House]:
        """
        Retrieve specific house by ID.

        If the house ID doesn't exist, return None.
        """
        try:
            return self.houses[house_id]
        except KeyError:
            return None

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate average house price, optionally filtered by bedrooms.

        If no houses match the criteria, return 0.
        """

        filtered_houses: Dict[str, List(float)] = dict()
        average = 0

        for house in self.houses.values():
            if bedrooms is not None:
                bedroom_list_key = str(house.bedrooms) + "-bedrooms"

                # Create a new key if it doesn't exist, and initialize with an empty list
                if bedroom_list_key not in filtered_houses:
                    filtered_houses[bedroom_list_key] = []

                # Append the house price to the corresponding key's list
                filtered_houses[bedroom_list_key].append(house.price)
            else:
                # Handle the "no bedroom amount specified" case
                if "-1" not in filtered_houses:
                    filtered_houses["-1"] = []
                filtered_houses["-1"].append(house.price)

        if not filtered_houses:  # Check if list is empty
            return 0
        else:
            for bedroom_amount in filtered_houses.keys():
                average = mean(filtered_houses[bedroom_amount])

        return average  # Calculate and return the average price

    def get_house_segment(self, house: House,
                          annual_income: float) -> Optional[Segment]:
        if house.is_new_construction(
        ) and house.quality_score == QualityScore.EXCELLENT:
            return Segment.FANCY

        elif house.price <= self.calculate_average_price():
            return Segment.AVERAGE

        elif house.calculate_price_per_square_foot() < (annual_income / 12):
            return Segment.OPTIMIZER

        else:
            print("House does not belong to any known segment")
            return None

    def get_houses_that_meet_requirements(
            self, max_price: float, segment: Segment,
            annual_income: float) -> Optional[List[House]]:
        """
        Filter houses based on buyer requirements.

        Implementation tips:
        - Consider multiple filtering criteria
        - Implement efficient filtering
        - Handle case when no houses match
        """

        if segment is None:
            raise ValueError("Invalid segment provided!")

        filtered_houses = list()

        for house in self.houses.values():
            house_segment = self.get_house_segment(house, annual_income)
            
            if house.price <= max_price and house_segment == segment:

                filtered_houses.append(house)

        if len(filtered_houses) > 0:
            return filtered_houses
        else:
            print(
                f"No houses marched the specified criteria: \nMax price: {max_price} \nSegment: {segment}"
            )
            return None  # Return None if no houses matched the criteria
