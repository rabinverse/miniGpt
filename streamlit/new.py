import streamlit as st


st.title(
    "Text Generator (MiniGpt)",
    help="Imitates the text paragraph (The input paragraph provided)",
)

# prompt
# prompt = st.chat_input("Enter starting text:",
#     help="Enter any letter or words or paragraph",
# )


prompt = st.chat_input(placeholder="Enter any word/letter/paragraph")
max_tokens = st.slider(
    "Max new tokens to generate:",
    min_value=50,
    max_value=950,
    value=250,
    help="This project uses character level tokenization",
)
st.header("Model Response: ")