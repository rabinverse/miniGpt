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

st.title(
    "Text Generator (MiniGpt)",
    help="Imitates the text paragraph (The input paragraph provided)",
)

# prompt
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
st.session_state.prompt = st.text_area(
    "Enter starting text:",
    value=st.session_state.prompt,
    placeholder="Sunday",
    help="Enter any letter or words or paragraph",
)

#  tokens slider
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 500
st.session_state.max_tokens = st.slider(
    "Max new tokens to generate:", 50, 950, st.session_state.max_tokens,help="This project uses character level tokenization"
)

# Generated output
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

# Generate button
if st.button("Generate"):
    if st.session_state.prompt.strip():
        with st.spinner("Generating text..."):
            idx = torch.tensor(
                [encode_txt(st.session_state.prompt)], dtype=torch.long, device=device
            )
            with torch.no_grad():
                generated_idx = st.session_state.model.generate(
                    idx, max_new_tokens=st.session_state.max_tokens
                )
            st.session_state.generated_text = decode_txt(generated_idx[0].tolist())
    else:
        st.warning("Please enter any English text.")

# Display generated text
if st.session_state.generated_text:
    st.text_area(
        "Generated Text:",
        value=st.session_state.generated_text,
        height=400,
        key="output_text_area",
    )
