import pickle
import pandas as pd
import numpy as np
import requests

import streamlit as st

from openai import OpenAI

# Nutrients of interest
nutrients_of_interest = {
    "Engery (kcal)": 1008,
    "Total Fat": 1004,
    "Saturated Fat": 1258,
    "Monounsaturated Fat": 1292,
    "Polyunsaturated Fat": 1293,
    "Cholesterol": 1253,
    "Carbohydrate": 1005,
    "Fiber": 1079,
    "Sugars": 2000,
    "Protein": 1003,
    "Calcium": 1087,
    "Iron": 1089,
    "Magnesium": 1090,
    "Phosphorus": 1091,
    "Potassium": 1092,
    "Sodium": 1093,
    "Zinc": 1095,
    "Copper": 1098,
    "Manganese": 1101,
    "Selenium": 1103,
    "Vitamin A": 1106,
    "Thiamin (B1)": 1165,
    "Riboflavin (B2)": 1166,
    "Niacin (B3)": 1167,
    "Pantothenic Acid (B5)": 1170,
    "Vitamin B6": 1175,
    "Folate (B9)": 1177,
    "Vitamin B12": 1178,
    "Vitamin C": 1162,
    "Vitamin D": 1110,
    "Vitamin E": 1109,
    "Vitamin K": 1185,
    "Choline": 1180,
}

daily_recommendations = {
    "Engery (kcal)": "~2000 per day",
    "Total Fat": "/",
    "Saturated Fat": "/",
    "Monounsaturated Fat": "/",
    "Polyunsaturated Fat": "/",
    "Cholesterol": "/",
    "Carbohydrate": "/",
    "Fiber": "~30g per day",
    "Sugars": "< 20g per day",
    "Protein": ">0.8g per kg body weight, per day",
    "Calcium": 1000,
    "Iron": 8,
    "Magnesium": 400,
    "Phosphorus": 700,
    "Potassium": 3000,
    "Sodium": 1500,
    "Zinc": 11,
    "Copper": 0.9,
    "Manganese": 2.3,
    "Selenium": 55,
    "Vitamin A": 900,
    "Thiamin (B1)": 1.2,
    "Riboflavin (B2)": 1.3,
    "Niacin (B3)": 16,
    "Pantothenic Acid (B5)": 5,
    "Vitamin B6": 1.3,
    "Folate (B9)": 400,
    "Vitamin B12": 2.4,
    "Vitamin C": 90,
    "Vitamin D": 600,
    "Vitamin E": 15,
    "Vitamin K": 120,
    "Choline": 550,
}


def make_food_dict():
    food_numbers = pd.read_csv("data/food.csv")
    # drop some food categories
    food_numbers = food_numbers[
        ~food_numbers["food_category_id"].isin([26, 27, 25, 24, 21, 22, 3])
    ]
    # keep only fdc_id and description
    food_numbers = food_numbers[["fdc_id", "description"]]
    # remove duplicates
    food_numbers = food_numbers.drop_duplicates()
    # convert to dictionary
    food_numbers = food_numbers.set_index("fdc_id").to_dict()["description"]

    with open("data/food_dict.pkl", "wb") as f:
        pickle.dump(food_numbers, f)


def make_nutrient_dict():
    nut_numbers = pd.read_csv("data/nutrient.csv")
    # keep only fdc_id and description
    nut_numbers = nut_numbers[["id", "name"]]
    # remove duplicates
    nut_numbers = nut_numbers.drop_duplicates()
    # convert to dictionary
    nut_numbers = nut_numbers.set_index("id").to_dict()["name"]

    with open("data/nutrient_dict.pkl", "wb") as f:
        pickle.dump(nut_numbers, f)


def load_food_dict():
    with open("data/food_dict.pkl", "rb") as f:
        return pickle.load(f)


def load_nutrient_dict():
    with open("data/nutrient_dict.pkl", "rb") as f:
        return pickle.load(f)


def get_client():
    return OpenAI(api_key=st.secrets["openai"])


def get_embbedding(food_item, client, dimensions=500):
    response = client.embeddings.create(
        input=food_item, model="text-embedding-3-small", dimensions=dimensions
    )

    return np.array(response.data[0].embedding)


def get_all_embeddings(data, client, batch_size=200, dimensions=500):
    embeddings = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        embeddings.extend(
            client.embeddings.create(
                input=batch, model="text-embedding-3-small", dimensions=dimensions
            ).data
        )
    return embeddings


def load_food_embeddings():
    all_embeddings = np.load("data/food_embeddings_500.npy")
    return all_embeddings


def get_closest_embeddings(embedding, food_dict, n):
    all_embeddings = load_food_embeddings()
    similarity = np.dot(all_embeddings, embedding)

    top_n = np.argsort(similarity)[::-1][:n]
    # sort by decreasing similarity
    return [list(food_dict.values())[i] for i in top_n]


def get_nutrient_data(food_id, nutrients):
    url = f"https://api.nal.usda.gov/fdc/v1/food/{food_id}"
    params = {"api_key": st.secrets["fooddata"]}
    response = requests.get(url, params=params).json()

    nutrient_data = {}
    for nutrient in nutrients:
        nutrient_data[nutrient] = None  # Default to None if nutrient not found
        if "foodNutrients" in response:
            for item in response["foodNutrients"]:
                if "nutrient" in item and item["nutrient"]["id"] == nutrients[nutrient]:
                    nutrient_data[nutrient] = item.get("amount")
                    break

    return nutrient_data


@st.cache_resource
def get_nutrient_table(food_df, food_dict, timeframe):
    nutrient_table = pd.DataFrame(
        index=food_df["Food"], columns=nutrients_of_interest.keys()
    )
    for name in food_df["Food"]:
        # get id in food dict
        id = [key for key, val in food_dict.items() if val == name][0]
        nutrients = get_nutrient_data(id, nutrients_of_interest)
        nutrient_table.loc[name] = nutrients

    # scale every item to food amount
    amount_col = (
        food_df["Weekly Amount (g)"]
        if timeframe == "Week"
        else food_df["Daily Amount (g)"]
    )
    for name, amount in zip(food_df["Food"], amount_col):
        nutrient_table.loc[name] = nutrient_table.loc[name] * amount / 100
    return nutrient_table


@st.cache_resource
def get_nutrient_summary(nutrient_table, timeframe):
    total_nutrition = nutrient_table.sum()
    total_nutrition_df = pd.DataFrame(
        columns=["Total from food", f"Requirement ({timeframe})"],
        index=total_nutrition.index,
    )
    for nutrient in total_nutrition.index:
        total_nutrition_df.loc[nutrient, "Total from food"] = format(
            total_nutrition[nutrient], ".2f"
        )
        if isinstance(daily_recommendations[nutrient], str):
            total_nutrition_df.loc[nutrient, f"Requirement ({timeframe})"] = (
                daily_recommendations[nutrient]
            )
        else:
            if timeframe == "Day":
                total_nutrition_df.loc[nutrient, f"Requirement ({timeframe})"] = (
                    daily_recommendations[nutrient]
                )
            else:
                total_nutrition_df.loc[nutrient, f"Requirement ({timeframe})"] = (
                    daily_recommendations[nutrient] * 7
                )

            if isinstance(
                total_nutrition_df.loc[nutrient, f"Requirement ({timeframe})"], int
            ):
                total_nutrition_df.loc[nutrient, f"Requirement ({timeframe})"] = format(
                    total_nutrition_df.loc[nutrient, f"Requirement ({timeframe})"],
                    ".0f",
                )
            else:
                total_nutrition_df.loc[nutrient, f"Requirement ({timeframe})"] = format(
                    total_nutrition_df.loc[nutrient, f"Requirement ({timeframe})"],
                    ".2f",
                )
    return total_nutrition_df
