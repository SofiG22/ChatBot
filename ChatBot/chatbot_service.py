from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Cargar modelo de QA
print("Cargando modelo de QA...")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", max_answer_length=100000)
print("Modelo de QA cargado.")

@app.route("/chat", methods=["POST"])
def chat():
    """Procesa la pregunta del usuario con el contexto proporcionado."""
    datos = request.get_json()
    
    pregunta = datos.get("pregunta", "").strip()
    contexto = datos.get("contexto", "").strip() 

    if not pregunta or not contexto:
        return jsonify({"error": "Faltan datos en la solicitud. Se requieren 'pregunta' y 'contexto'."}), 400

    try:
        respuesta = qa_pipeline({"question": pregunta, "context": contexto})
        return jsonify({"respuesta": respuesta["answer"]})
    except Exception as e:
        print(f"Error procesando la pregunta: {e}")
        return jsonify({"error": "No se pudo generar una respuesta."}), 500

if __name__ == "__main__":
    print("Chatbot Service iniciado en http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
