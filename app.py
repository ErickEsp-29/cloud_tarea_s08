import streamlit as st
import pymongo
from google import genai
from google.genai import types

# =======================
# CONFIGURACIÓN
# =======================

st.set_page_config(
    page_title="BasketBot 🏀",
    page_icon="🏀",
    layout="wide"
)

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

# =======================
# ESTILOS PERSONALIZADOS
# =======================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

.stApp {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #111827 100%
    );
}

h1, h2, h3 {
    color: #f97316 !important;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

[data-testid="stChatInput"] {
    border-radius: 15px;
}

.basket-card {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #334155;
    margin-bottom: 10px;
}

.small-text {
    color: #cbd5e1;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =======================
# CLIENTES
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():

    client = pymongo.MongoClient(MONGODB_URI)

    db = client["basket_chatbot_db"]

    return db["basket_pdf_vectors"]

client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# FUNCIONES
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
# RESPUESTA GEMINI
# =======================

def generar_respuesta(pregunta: str, contextos: list[dict]) -> str:

    contexto = "\n\n".join([c["texto"] for c in contextos])

    prompt = f"""
Eres BasketBot 🏀, un asistente experto en baloncesto.

Usa ÚNICAMENTE la información del contexto.

Si la respuesta no está en el contexto, indícalo claramente.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

Responde en español, de manera clara y amigable.
"""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text

# =======================
# FUNCIÓN PRINCIPAL CHAT
# =======================

def procesar_pregunta(pregunta):

    # Mostrar mensaje usuario
    st.chat_message("user").write(pregunta)

    st.session_state.historial.append({
        "rol": "usuario",
        "texto": pregunta
    })

    # Respuesta bot
    with st.chat_message("assistant"):

        with st.spinner("🏀 Analizando jugada..."):

            try:

                emb = crear_embedding(pregunta)

                similares = buscar_similares(emb, k=5)

                if not similares:

                    respuesta = (
                        "No encontré información sobre eso "
                        "en el PDF de baloncesto."
                    )

                else:

                    respuesta = generar_respuesta(
                        pregunta,
                        similares
                    )

            except Exception as e:

                respuesta = f"⚠️ Error: {e}"

        st.write(respuesta)

        # Fragmentos recuperados
        if 'similares' in locals() and similares:

            with st.expander("📚 Ver fragmentos utilizados"):

                for i, c in enumerate(similares, 1):

                    st.markdown(
                        f"### 🏀 Fragmento {i}"
                    )

                    st.caption(
                        f"Score: {c['score']:.4f}"
                    )

                    st.write(
                        c["texto"][:500] +
                        ("..." if len(c["texto"]) > 500 else "")
                    )

                    st.divider()

    st.session_state.historial.append({
        "rol": "bot",
        "texto": respuesta
    })

# =======================
# TÍTULO
# =======================

st.title("🏀 BasketBot")
st.markdown("""
<div class="basket-card">

### Chatbot inteligente sobre baloncesto

Pregunta sobre:
- reglas
- posiciones
- tácticas
- fundamentos
- historia del basket

El bot responderá usando únicamente el contenido de tu PDF.

</div>
""", unsafe_allow_html=True)

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
# PREGUNTAS SUGERIDAS
# =======================

st.subheader("🔥 Preguntas rápidas")

col1, col2, col3 = st.columns(3)

preguntas_sugeridas = [
    "¿Cuáles son las reglas básicas del baloncesto?",
    "¿Cuántos jugadores hay en un equipo?",
    "¿Qué posiciones existen en basket?",
    "¿Qué es un triple?",
    "¿Cómo se gana un partido?",
    "¿Qué faltas existen en baloncesto?"
]

botones = []

with col1:
    botones.append(
        st.button(
            preguntas_sugeridas[0],
            use_container_width=True
        )
    )

    botones.append(
        st.button(
            preguntas_sugeridas[1],
            use_container_width=True
        )
    )

with col2:
    botones.append(
        st.button(
            preguntas_sugeridas[2],
            use_container_width=True
        )
    )

    botones.append(
        st.button(
            preguntas_sugeridas[3],
            use_container_width=True
        )
    )

with col3:
    botones.append(
        st.button(
            preguntas_sugeridas[4],
            use_container_width=True
        )
    )

    botones.append(
        st.button(
            preguntas_sugeridas[5],
            use_container_width=True
        )
    )

# Detectar clic en botones
for i, clicked in enumerate(botones):

    if clicked:

        pregunta_boton = preguntas_sugeridas[i]

        procesar_pregunta(pregunta_boton)

# =======================
# INPUT MANUAL
# =======================

pregunta = st.chat_input(
    "Escribe una pregunta sobre baloncesto..."
)

if pregunta:

    procesar_pregunta(pregunta)
