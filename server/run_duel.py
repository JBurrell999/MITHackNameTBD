import streamlit as st
from models.controller import determine_winner
from models.dream_env import get_environment
from models.mdn_rnn import simulate_rnn
from models.vae import encode_players
from utils.clamp import clamp_stats
from utils.exploit_checks import check_exploit

st.title("Advanced Duel Simulator")

player1 = st.text_input("Enter name for Player 1:")
player2 = st.text_input("Enter name for Player 2:")

if st.button("Start Duel"):
    if player1 and player2:
        # Example usage of other modules
        env = get_environment()
        p1_encoded = encode_players(player1)
        p2_encoded = encode_players(player2)
        p1_stats = clamp_stats(p1_encoded)
        p2_stats = clamp_stats(p2_encoded)
        rnn_result = simulate_rnn(p1_stats, p2_stats, env)
        exploit_flag = check_exploit(p1_stats, p2_stats)
        winner = determine_winner(rnn_result, exploit_flag)
        st.success(f"The winner is: {winner}!")
    else:
        st.warning("Please enter names for both players.")
