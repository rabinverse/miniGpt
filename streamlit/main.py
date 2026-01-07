import torch
import streamlit as st
from gpt import BigramLanguageModel, decode_txt, encode_txt, device

# Initialize model
if "model" not in st.session_state:
    model = BigramLanguageModel()
    model.load_state_dict(
        torch.load(
            "./dataset_research_paper_docs/gpt_model_weights.pt", map_location=device
        )
    )
    model.to(device)
    model.eval()
    st.session_state.model = model

st.set_page_config(
    page_title="MiniGPT Text Generator",
    layout="centered"
)

st.title(
    "Text Generator (MiniGpt)",
    help="Imitates the text paragraph (The input paragraph provided)",
)

# prompt
prompt = st.text_area(
    "Enter starting text:",
    placeholder="Sunday",
    help="Enter any letter or words or paragraph",
)
#  tokens slider

max_tokens = st.slider(
    "Max new tokens to generate:",
    min_value=50,
    max_value=950,
    value=250,
    help="This project uses character level tokenization",
)


# Generate button
if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Generating text..."):
            try:
                idx = torch.tensor(
                    [encode_txt(prompt)], dtype=torch.long, device=device
                )
            except KeyError as e:
                st.error("Please enter English text only. Other scripts are not supported.")
                st.divider()
                st.stop()
            with torch.no_grad():
                generated_idx = st.session_state.model.generate(
                    idx, max_new_tokens=max_tokens
                )
            generated_text = decode_txt(generated_idx[0].tolist())
            st.divider()
            st.header("Model Response: ")
            st.code(generated_text, language="text")

    else:
        st.warning("Please enter any English text.")
