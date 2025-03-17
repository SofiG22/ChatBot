from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)


CHATBOT_SERVICE_URL = "http://localhost:5001"

@app.route("/chatbot", methods=["POST"])
def chatbot():
    
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

if __name__ == "__main__":
    print("🚀 API Gateway iniciado en http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
