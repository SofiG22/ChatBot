from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# URLs de los servicios
CHATBOT_SERVICE_URL = "http://localhost:5001"
GENDER_CLASSIFICATION_URL = "http://localhost:5002"
EMOTION_ANALYSIS_URL = "http://localhost:5003" 

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Proxy para el servicio de chatbot."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Faltan datos en la solicitud."}), 400
    
    try:
        response = requests.post(f"{CHATBOT_SERVICE_URL}/chat", json=data)
        if response.status_code != 200 or not response.text.strip():
            return jsonify({"error": "No se obtuvo respuesta válida del chatbot."}), 500
        
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al comunicarse con el chatbot: {e}")
        return jsonify({"error": "No se pudo conectar con el chatbot."}), 500

@app.route("/clasificar", methods=["POST"])
def clasificar_genero():
    """Proxy para el servicio de clasificación de imágenes."""
    
    # Caso 1: Si se envió un archivo de imagen
    if request.files and 'imagen' in request.files:
        try:
            # Reenviamos el archivo de imagen al servicio
            archivo = request.files['imagen']
            archivos = {'imagen': (archivo.filename, archivo.read(), archivo.content_type)}
            response = requests.post(f"{GENDER_CLASSIFICATION_URL}/clasificar", files=archivos)
            return jsonify(response.json()), response.status_code
        except requests.exceptions.RequestException as e:
            print(f"❌ Error al comunicarse con el servicio de clasificación: {e}")
            return jsonify({"error": "No se pudo conectar con el servicio de clasificación."}), 500
    
    # Caso 2: Si se envió JSON (imagen en base64 o URL)
    elif request.is_json:
        try:
            # Reenviamos el JSON tal cual
            data = request.get_json()
            response = requests.post(f"{GENDER_CLASSIFICATION_URL}/clasificar", json=data)
            return jsonify(response.json()), response.status_code
        except requests.exceptions.RequestException as e:
            print(f"❌ Error al comunicarse con el servicio de clasificación: {e}")
            return jsonify({"error": "No se pudo conectar con el servicio de clasificación."}), 500
    
    # Caso 3: Si no hay ni archivos ni JSON
    else:
        return jsonify({"error": "Formato de solicitud no válido. Envíe un archivo de imagen o JSON con la imagen en base64."}), 400

@app.route("/detectar-emocion", methods=["POST"])
def detectar_emocion():
    """Proxy para el servicio de análisis de emociones en texto."""
    if not request.is_json:
        return jsonify({"error": "La solicitud debe ser en formato JSON."}), 400
    
    data = request.get_json()
    if not data or "texto" not in data:
        return jsonify({"error": "Faltan datos en la solicitud. Se requiere el campo 'texto'."}), 400
    
    try:
        response = requests.post(f"{EMOTION_ANALYSIS_URL}/detectar-emocion", json=data)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al comunicarse con el servicio de análisis de emociones: {e}")
        return jsonify({"error": "No se pudo conectar con el servicio de análisis de emociones."}), 500

@app.route("/batch-emociones", methods=["POST"])
def batch_emociones():
    """Proxy para el análisis de múltiples textos."""
    if not request.is_json:
        return jsonify({"error": "La solicitud debe ser en formato JSON."}), 400
    
    data = request.get_json()
    if not data or "textos" not in data or not isinstance(data["textos"], list):
        return jsonify({"error": "Faltan datos en la solicitud. Se requiere el campo 'textos' como lista."}), 400
    
    try:
        response = requests.post(f"{EMOTION_ANALYSIS_URL}/batch", json=data)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al comunicarse con el servicio de análisis de emociones: {e}")
        return jsonify({"error": "No se pudo conectar con el servicio de análisis de emociones."}), 500

@app.route("/estado-emocion", methods=["GET"])
def estado_emocion():
    """Proxy para verificar el estado del servicio de emociones."""
    try:
        response = requests.get(f"{EMOTION_ANALYSIS_URL}/estado")
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al comunicarse con el servicio de análisis de emociones: {e}")
        return jsonify({"error": "No se pudo conectar con el servicio de análisis de emociones."}), 500

if __name__ == "__main__":
    print("🚀 API Gateway iniciado en http://localhost:5000")
    print("📌 Endpoints disponibles:")
    print("   - /chatbot: Redirección al servicio de chatbot")
    print("   - /clasificar: Redirección al servicio de clasificación de imágenes")
    print("   - /detectar-emocion: Redirección al servicio de análisis de emociones")
    print("   - /batch-emociones: Redirección para análisis de múltiples textos")
    print("   - /estado-emocion: Verifica el estado del servicio de emociones")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)