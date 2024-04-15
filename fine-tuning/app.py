import streamlit as st
from inference import get_top_k
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

model_name = "/Users/amitej/amitejmehta/models/gpt2"
if 'tokenizer' not in st.session_state:
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(model_name)
if 'model' not in st.session_state:
    st.session_state['model'] = AutoModelForCausalLM.from_pretrained(
        model_name)

tokenizer = st.session_state['tokenizer']
model = st.session_state['model']

st.title("Token Distribution Visualizer")


prompt = st.text_input("Talk to GPT-2")
input = tokenizer(prompt, return_tensors="pt")


def show_plot(sampling, i, temperature):
    x, y = get_top_k(input, model, tokenizer, sampling=sampling, i=i,
                     k=10, temperature=temperature)
    fig, ax = plt.subplots()  # Use fig and ax for better control
    ax.bar(x, y)
    ax.set_xlabel('Top 10 Tokens')
    ax.set_ylabel('Probability')
    ax.set_title("Probability Distribution of Top 10 Tokens")
    ax.set_xticklabels(x, rotation=-45)
    st.pyplot(fig)  # Use Streamlit's function to show the plot


sampling = st.selectbox("Select Sampling Method for Generation", [
    "greedy", "top_k", "top_p"])
i = st.slider("Select a Token", 0, 100)
temperature = st.slider("Change the Temperature", 1, 10)


if prompt:
    show_plot(sampling, i, temperature)

st.text("What happens to the distribution as temperature increases?")
