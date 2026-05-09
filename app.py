import streamlit as st
import pymongo
from google import genai
from google.genai import types

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

# =======================
# CLIENTES
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():

    client = pymongo.MongoClient(MONGODB_URI)

    # Base de datos y colección de basket
    db = client["basket_chatbot_db"]

    return db["basket_pdf_vectors"]

client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# CREAR EMBEDDING
# =======================

def crear_embedding(texto: str):

    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    return response.embeddings[0].values

# =======================
# BÚSQUEDA VECTORIAL
# =======================

def buscar_similares(embedding, k=5):

    pipeline = [
        {
            "$vectorSearch": {
                "index": "basket_vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "tema": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(collection.aggregate(pipeline))

# =======================
# GENERAR RESPUESTA
# =======================

def generar_respuesta(pregunta: str, contextos: list[dict]) -> str:

    contexto = "\n\n".join([c["texto"] for c in contextos])

    prompt = f"""
Eres un experto en baloncesto.

Usa EXCLUSIVAMENTE la información del contexto para responder.

Si la respuesta no aparece en el contexto, dilo claramente.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

Responde de manera clara, breve y en español.
"""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text

# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(
    page_title="Chatbot de Basket",
    page_icon="🏀"
)

st.title("🏀 Chatbot de Basket con Gemini + MongoDB")

st.markdown(
    "Haz preguntas sobre el PDF de baloncesto."
)

# =======================
# HISTORIAL
# =======================

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar historial
for msg in st.session_state.historial:

    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])

    else:
        st.chat_message("assistant").write(msg["texto"])

# =======================
# INPUT USUARIO
# =======================

pregunta = st.chat_input(
    "Pregunta algo sobre baloncesto..."
)

if pregunta:

    # Mostrar pregunta usuario
    st.chat_message("user").write(pregunta)

    st.session_state.historial.append({
        "rol": "usuario",
        "texto": pregunta
    })

    # Respuesta bot
    with st.chat_message("assistant"):

        with st.spinner("🏀 Buscando información..."):

            try:

                # Embedding de la pregunta
                emb = crear_embedding(pregunta)

                # Buscar fragmentos similares
                similares = buscar_similares(emb, k=5)

                if not similares:

                    respuesta = (
                        "No encontré información relevante "
                        "sobre basket en el PDF."
                    )

                else:

                    respuesta = generar_respuesta(
                        pregunta,
                        similares
                    )

            except Exception as e:

                respuesta = f"⚠️ Error: {e}"

        st.write(respuesta)

        # Mostrar fragmentos encontrados
        if 'similares' in locals() and similares:

            with st.expander("🔍 Fragmentos encontrados"):

                for i, c in enumerate(similares, 1):

                    st.markdown(
                        f"**Fragmento {i}** "
                        f"(score: `{c['score']:.4f}`)"
                    )

                    st.write(
                        c["texto"][:500] +
                        ("..." if len(c["texto"]) > 500 else "")
                    )

                    st.divider()

    # Guardar respuesta
    st.session_state.historial.append({
        "rol": "bot",
        "texto": respuesta
    })
