import streamlit as st
from inference import get_top_k
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

model_name = "/Users/amitej/amitejmehta/models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


prompt = st.text_input("Talk to GPT-2")
input = tokenizer(prompt, return_tensors="pt")


def show_plot(i, temperature):
    x, y = get_top_k(input, model, tokenizer, k=10,
                     i=i, temperature=temperature, sampling='greedy')
    fig, ax = plt.subplots()  # Use fig and ax for better control
    ax.bar(x, y)
    ax.set_xlabel('Top 10 Tokens')
    ax.set_ylabel('Probability')
    ax.set_title("Probability Distribution of Top 10 Tokens")
    ax.set_xticklabels(x, rotation=-45)
    st.pyplot(fig)  # Use Streamlit's function to show the plot


i = st.slider("Select a Token", 1, 25)
temperature = st.slider("Change the Temperature", 1, 10)

if prompt:
    show_plot(i, temperature)


st.title("Token Distribution Visualizer")
