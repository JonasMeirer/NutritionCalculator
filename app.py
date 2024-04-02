import streamlit as st
import pandas as pd

from utils import (
    load_food_dict,
    load_nutrient_dict,
    get_client,
    get_embbedding,
    load_food_embeddings,
    get_closest_embeddings,
    get_nutrient_table,
    get_nutrient_summary,
)

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

st.set_page_config(layout="wide")

with open("data/user.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout()
    # state variables
    if "food_df" not in st.session_state:
        st.session_state.food_df = None

    if "timeframe" not in st.session_state:
        st.session_state.timeframe = "Week"

    def change_timeframe():
        if st.session_state.food_df is not None:
            if st.session_state.timeframe == "Day":
                st.session_state.food_df = st.session_state.food_df.rename(
                    columns={"Daily Amount (g)": "Weekly Amount (g)"}
                )
                st.session_state.food_df["Weekly Amount (g)"] = (
                    st.session_state.food_df["Weekly Amount (g)"] * 7
                )
            else:
                st.session_state.food_df = st.session_state.food_df.rename(
                    columns={"Weekly Amount (g)": "Daily Amount (g)"}
                )
                st.session_state.food_df["Daily Amount (g)"] = (
                    st.session_state.food_df["Daily Amount (g)"] / 7
                )

    def add_food_item(food_item):
        if st.session_state.food_df is None:
            if st.session_state.timeframe == "Day":
                st.session_state.food_df = pd.DataFrame(
                    {"Food": [food_item], "Daily Amount (g)": [0]}
                )
            else:
                st.session_state.food_df = pd.DataFrame(
                    {"Food": [food_item], "Weekly Amount (g)": [0]}
                )

        else:
            if food_item not in st.session_state.food_df["Food"].values:
                if st.session_state.timeframe == "Day":
                    st.session_state.food_df = pd.concat(
                        [
                            st.session_state.food_df,
                            pd.DataFrame(
                                {"Food": [food_item], "Daily Amount (g)": [0]}
                            ),
                        ],
                        axis=0,
                    )
                else:
                    st.session_state.food_df = pd.concat(
                        [
                            st.session_state.food_df,
                            pd.DataFrame(
                                {"Food": [food_item], "Weekly Amount (g)": [0]}
                            ),
                        ],
                        axis=0,
                    )

    st.title("goodlife")

    st.header("Understanding your nutrition")

    client = get_client()
    all_embeddings = load_food_embeddings()

    food_dict = load_food_dict()
    nutrient_dict = load_nutrient_dict()

    # text input
    st.header("Step 1: Add the foods you eat")
    st.write(
        """Of course it's not possible to list it all. 
        But just listing the foods you buy in the supermarket for a week 
        will already give you a good minimum nutritional profile. 
        """
    )
    with st.expander("Collect all food items", expanded=True):
        st.subheader("Search for a food item")
        mode = st.checkbox("Pro Version")
        food_item = st.text_input("Enter a food item")

        if food_item:
            if not mode:
                # search for food item in food dict, allowing for spelling errors
                search_results = [
                    food_dict[key]
                    for key, val in food_dict.items()
                    if food_item.lower() in val.lower()
                ]
                # sort them by decreasing length
                search_results = sorted(search_results, key=len, reverse=False)
            else:
                embedding = get_embbedding(food_item, client)
                search_results = get_closest_embeddings(all_embeddings, embedding, 5)

            st.subheader("Results")
            if len(search_results) == 0:
                st.write("No results found")
            else:
                selected_food = st.selectbox("Select a food item", search_results, None)
                if selected_food:
                    st.button(
                        "Add food item", on_click=add_food_item, args=(selected_food,)
                    )

    if st.session_state.food_df is not None:
        st.header("Step 2: Provide food amounts")
        st.session_state.timeframe = st.radio(
            "Interval",
            ["Day", "Week"],
            index=1,
            on_change=change_timeframe,
            horizontal=True,
        )

        st.session_state.food_df = st.data_editor(
            st.session_state.food_df,
            hide_index=True,
            disabled=["Food"],
            column_config={
                st.session_state.food_df.columns[1]: st.column_config.NumberColumn(
                    format="%.0f"
                )
            },
        )

        st.header("Step 3: Nutrition Analysis")
        if st.button("Run Analysis"):
            with st.expander("Individual food analysis", expanded=False):
                nutrition_df = get_nutrient_table(
                    st.session_state.food_df, food_dict, st.session_state.timeframe
                )
                st.dataframe(nutrition_df, hide_index=False, use_container_width=True)

            with st.expander("Total nutrition analysis", expanded=False):
                total_nutrition = get_nutrient_summary(
                    nutrition_df, st.session_state.timeframe
                )
                st.dataframe(total_nutrition, use_container_width=True)


elif st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"] is None:
    st.warning("Please enter your username and password")
