from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

print("⏳ Cargando modelo de clasificación de imágenes...")

# Cargar el modelo y el procesador de características de Hugging Face
# Usamos un modelo ViT preentrenado en ImageNet
model_name = "google/vit-base-patch16-224"
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    print(f"✅ Modelo {model_name} cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    print("⚠️ Intentando con un modelo más pequeño...")
    # Intentar con un modelo más pequeño como alternativa
    model_name = "microsoft/resnet-18"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    print(f"✅ Modelo alternativo {model_name} cargado correctamente")

def classify_image(img):
    """Clasifica una imagen usando el modelo preentrenado"""
    try:
        # Preprocesar la imagen
        inputs = feature_extractor(images=img, return_tensors="pt")
        
        # Realizar la predicción
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Obtener las predicciones principales
        predicted_class_ids = logits.argmax(-1).tolist()
        probabilities = logits.softmax(dim=-1)[0]
        
        # Extraer top 3 predicciones
        top_3_indices = probabilities.argsort(descending=True)[:3].tolist()
        
        results = []
        for idx in top_3_indices:
            # Obtener la etiqueta y la probabilidad
            label = model.config.id2label[idx]
            probability = float(probabilities[idx].item() * 100)
            results.append({
                "label": label,
                "probability": probability
            })
        
        return {"success": True, "predictions": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

def process_image_from_url(img_url):
    """Procesa una imagen desde una URL"""
    try:
        response = requests.get(img_url)
        if response.status_code != 200:
            return {"success": False, "error": f"Error descargando imagen: Código de estado {response.status_code}"}
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return classify_image(img)
    except Exception as e:
        return {"success": False, "error": f"Error procesando imagen desde URL: {str(e)}"}

def process_image_from_file(file_content):
    """Procesa una imagen desde contenido de archivo"""
    try:
        img = Image.open(BytesIO(file_content)).convert('RGB')
        return classify_image(img)
    except Exception as e:
        return {"success": False, "error": f"Error procesando archivo de imagen: {str(e)}"}

def process_image_from_base64(base64_string):
    """Procesa una imagen desde una cadena base64"""
    try:
        # Eliminar el prefijo si existe (como "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decodificar la cadena base64
        img_data = base64.b64decode(base64_string)
        
        return process_image_from_file(img_data)
    except Exception as e:
        return {"success": False, "error": f"Error procesando imagen base64: {str(e)}"}

@app.route('/clasificar', methods=['POST'])
def clasificar():
    """Endpoint principal para clasificación de imágenes"""
    
    # Caso 1: Si se envió un archivo de imagen
    if request.files and 'imagen' in request.files:
        try:
            archivo = request.files['imagen']
            file_content = archivo.read()
            result = process_image_from_file(file_content)
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": f"Error al procesar el archivo: {str(e)}"}), 400
    
    # Caso 2: Si se envió JSON
    elif request.is_json:
        data = request.get_json()
        
        # Si el JSON contiene una URL
        if 'url' in data:
            result = process_image_from_url(data['url'])
            return jsonify(result)
        
        # Si el JSON contiene base64
        elif 'imagen_base64' in data:
            result = process_image_from_base64(data['imagen_base64'])
            return jsonify(result)
        
        else:
            return jsonify({"success": False, "error": "Se esperaba 'url' o 'imagen_base64' en el JSON"}), 400
    
    # Caso 3: Si no hay ni archivos ni JSON válido
    else:
        return jsonify({
            "success": False, 
            "error": "Formato de solicitud no válido. Envíe un archivo de imagen, una URL o una imagen en base64."
        }), 400

if __name__ == '__main__':
    print("🚀 Servicio de clasificación de imágenes iniciado en http://localhost:5002")
    print("📌 Endpoint disponible:")
    print("   - /clasificar: Endpoint para clasificación de imágenes")
    app.run(host='0.0.0.0', port=5002, debug=True)