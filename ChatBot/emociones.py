from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

print("⏳ Cargando modelo de análisis de sentimientos...")

# Cargar el modelo y el tokenizador de Hugging Face
modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
try:
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModelForSequenceClassification.from_pretrained(modelo)
    print(f"✅ Modelo {modelo} cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    print("⚠️ Intentando con un modelo alternativo...")
    # Intentar con un modelo alternativo como fallback
    modelo = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModelForSequenceClassification.from_pretrained(modelo)
    print(f"✅ Modelo alternativo {modelo} cargado correctamente")

# Mapeo personalizado de emociones según el frontend
# El modelo BERT de sentimientos tiene 5 clases (0-4), las mapeamos a las emociones del frontend
emotion_mapping = {
    0: "Triste/a",        # Muy negativo -> Triste
    1: "Frustrado/a",     # Negativo -> Frustrado
    2: "Neutral/a",       # Neutral -> Neutral
    3: "Feliz/a",         # Positivo -> Feliz
    4: "Orgulloso/a"      # Muy positivo -> Orgulloso
}

# Mapeo secundario para emociones adicionales basadas en análisis de texto
secondary_emotions = {
    "error": "Confundido/a",
    "problema": "Frustrado/a", 
    "molest": "Enojado/a",
    "increíble": "Sorpresa/a",
    "amo": "Amor/a",
    "miedo": "Miedo/a",
    "confus": "Confundido/a",
    "verguenz": "Avergonzado/a",
    "orgull": "Orgulloso/a",
    "cansan": "Cansado/a",
    "nervios": "Nervioso/a",
    "frustra": "Frustrado/a",
    "aburr": "Aburrido/a",
    "relaja": "Relajado/a",
    "ansio": "Ansioso/a",
    "inspira": "Inspirado/a",
    "desinterés": "Desinteresado/a"
}

def get_secondary_emotion(texto, primary_emotion):
    """
    Analiza palabras clave en el texto para detectar emociones secundarias
    """
    texto_lower = texto.lower()
    
    # Si detectamos palabras clave específicas, asignamos emociones secundarias
    for keyword, emotion in secondary_emotions.items():
        if keyword in texto_lower:
            return emotion
    
    # Si no encontramos palabras clave, mantenemos la emoción primaria
    return primary_emotion

def detectar_emocion(texto):
    """
    Analiza el sentimiento de un texto usando el modelo preentrenado
    Retorna la emoción detectada y el nivel de confianza
    """
    try:
        # Preprocesar el texto
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
        
        # Realizar la predicción
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Obtener la predicción principal
        predicted_class = torch.argmax(logits).item()
        probabilities = torch.softmax(logits, dim=1)[0]
        
        # Obtener la emoción primaria según el modelo de sentimientos
        primary_emotion = emotion_mapping[predicted_class]
        
        # Buscar posibles emociones secundarias basadas en palabras clave
        detected_emotion = get_secondary_emotion(texto, primary_emotion)
        
        # Calcular confianza
        confidence = float(probabilities[predicted_class].item() * 100)
        
        # Obtener emoji correspondiente
        emoji = get_emoji_for_status(detected_emotion)
        
        return {
            "success": True, 
            "texto": texto,
            "emocion": detected_emotion,
            "emoji": emoji,
            "confianza": confidence,
            "emocion_base": primary_emotion
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_emoji_for_status(status):
    """Devuelve el emoji correspondiente a cada estado emocional"""
    emoji_map = {
        'Feliz/a': '😊', 
        'Triste/a': '😞',
        'Enojado/a': '😡', 
        'Neutral/a': '😐',
        'Sorpresa/a': '😲',
        'Amor/a': '❤️',
        'Miedo/a': '😨',
        'Confundido/a': '🤔',
        'Avergonzado/a': '😳',
        'Orgulloso/a': '😎',
        'Cansado/a': '😴',
        'Nervioso/a': '😬',
        'Frustrado/a': '😤',
        'Desinteresado/a': '😒',
        'Relajado/a': '😌',
        'Ansioso/a': '😟',
        'Aburrido/a': '😑',
        'Inspirado/a': '✨'
    }
    return emoji_map.get(status, '')

@app.route('/detectar-emocion', methods=['POST'])
def detectar_emocion_endpoint():
    """Endpoint para detectar la emoción en un texto."""
    
    # Verificar si se recibió JSON
    if not request.is_json:
        return jsonify({
            "success": False, 
            "error": "La solicitud debe ser en formato JSON."
        }), 400
    
    # Obtener el texto del JSON
    data = request.get_json()
    if not data or "texto" not in data:
        return jsonify({
            "success": False,
            "error": "Formato incorrecto. Se requiere el campo 'texto'."
        }), 400
    
    texto = data["texto"]
    
    # Analizar la emoción
    resultado = detectar_emocion(texto)
    return jsonify(resultado)

@app.route('/batch', methods=['POST'])
def batch_analysis():
    """Endpoint para analizar múltiples textos en una sola solicitud"""
    
    # Verificar si se recibió JSON
    if not request.is_json:
        return jsonify({
            "success": False, 
            "error": "La solicitud debe ser en formato JSON."
        }), 400
    
    # Obtener la lista de textos
    data = request.get_json()
    if not data or "textos" not in data or not isinstance(data["textos"], list):
        return jsonify({
            "success": False,
            "error": "Formato incorrecto. Se requiere el campo 'textos' como lista."
        }), 400
    
    textos = data["textos"]
    
    # Analizar cada texto
    resultados = []
    for texto in textos:
        resultado = detectar_emocion(texto)
        resultados.append(resultado)
    
    return jsonify({
        "success": True,
        "resultados": resultados
    })

@app.route('/estado', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servicio"""
    return jsonify({
        "status": "online",
        "model": modelo,
        "version": "1.0.0",
        "emociones_soportadas": list(set(emotion_mapping.values()))
    })

if __name__ == '__main__':
    print("🚀 Servicio de análisis de sentimientos iniciado en http://localhost:5003")
    print("📌 Endpoints disponibles:")
    print("   - /detectar-emocion: Detecta la emoción en un texto proporcionado")
    print("   - /batch: Analiza múltiples textos en una sola solicitud")
    print("   - /estado: Verifica el estado del servicio")
    app.run(host='0.0.0.0', port=5003, debug=True)