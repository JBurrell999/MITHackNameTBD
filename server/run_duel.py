import streamlit as st
import random

st.title("Simple Duel Simulator")

player1 = st.text_input("Enter name for Player 1:")
player2 = st.text_input("Enter name for Player 2:")

if st.button("Start  Duel"):
    if player1 and player2:
        winner = random.choice([player1, player2])
        st.success(f"The winner is: {winner}!")
    else:
        st.warning("Please enter names for both players.")
