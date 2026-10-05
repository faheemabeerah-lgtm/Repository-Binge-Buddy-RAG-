import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="BingeBuddy",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------
# Netflix-style CSS
# -----------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0b0b0f;
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #e50914;
    text-align: center;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #bbbbbb;
    font-size: 20px;
    margin-bottom: 35px;
}

.stTextInput > div > div > input {
    background-color: #18181f;
    color: white;
    border: 1px solid #444;
    border-radius: 10px;
    padding: 12px;
}

.stButton > button {
    width: 100%;
    background-color: #e50914;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-size: 17px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #b20710;
}

.recommendation {
    background-color: #18181f;
    padding: 25px;
    border-radius: 15px;
    margin-top: 25px;
    border: 1px solid #333;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Title
# -----------------------------
st.markdown(
    '<div class="main-title">🎬 BingeBuddy</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find your next obsession 🍿</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Load Groq API Key
# -----------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api_key)

except Exception:
    st.error("⚠️ GROQ_API_KEY is not configured.")
    st.info(
        "Add GROQ_API_KEY in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# -----------------------------
# Load Knowledge Base
# -----------------------------
@st.cache_resource
def load_data():

    # Read BingeBuddy knowledge base
    with open("binge_buddy.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # Split knowledge base into chunks
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    # Load Hugging Face embedding model
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create embeddings
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    # Create FAISS database
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return chunks, embedding_model, index


# -----------------------------
# Load everything
# -----------------------------
with st.spinner("🎬 Loading BingeBuddy..."):
    chunks, embedding_model, index = load_data()


# -----------------------------
# User Question
# -----------------------------
question = st.text_input(
    "🔍 What do you want to watch tonight?",
    placeholder="Example: I want a cute Korean romance..."
)


# -----------------------------
# Recommendation Button
# -----------------------------
if st.button("✨ Recommend Something"):

    if not question.strip():

        st.warning("Please enter what you want to watch.")

    else:

        with st.spinner("🍿 Finding your perfect match..."):

            # Convert question into embedding
            query_embedding = embedding_model.encode(
                [question],
                convert_to_numpy=True
            ).astype("float32")

            # Search FAISS
            distances, indices = index.search(
                query_embedding,
                3
            )

            # Get relevant chunks
            retrieved_chunks = [
                chunks[i]
                for i in indices[0]
            ]

            # Create context
            context = "\n\n".join(retrieved_chunks)

            # RAG prompt
            prompt = f"""
You are BingeBuddy, a friendly Netflix-style
movie and series recommendation assistant.

Use ONLY the information provided in the knowledge base.

BINGEBUDDY KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}

RULES:
- Recommend 2 to 4 shows when possible.
- Explain briefly why each recommendation matches.
- Mention genre, language, mood, or episode count when useful.
- Do not invent shows.
- Do not invent information.
- Do not reveal spoilers.
- Keep the answer friendly and concise.
- If the knowledge base does not contain enough information,
  clearly say that BingeBuddy does not have that information.
"""

            # Send request to Groq
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )

            answer = response.choices[0].message.content

        # Display result
        st.markdown(
            '<div class="recommendation">',
            unsafe_allow_html=True
        )

        st.subheader("🍿 BingeBuddy Recommends")

        st.write(answer)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )
