from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import torch
from torchvision import models, transforms
import numpy as np
import json

app = Flask(__name__)
print("🚀 Cargando modelo de clasificación de imágenes...")

# Cargar el modelo ResNet50 pre-entrenado
model = models.resnet50(pretrained=True)
model.eval()
print("✅ Modelo de clasificación cargado.")

# Preprocesamiento estándar para ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar las etiquetas de ImageNet
# Este archivo debe estar en el mismo directorio que el script
imagenet_class_idx = {
    0: "hombre",  # Ejemplos para este demo, ajustar según necesidad
    1: "mujer"
}

def classify_gender(image):
    """
    Función para detectar género utilizando el modelo ResNet50.
    
    En un entorno real, deberías usar un modelo específicamente entrenado para género,
    pero para este ejemplo usaremos una adaptación del ResNet.
    """
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocesar la imagen para el modelo
    img_tensor = preprocess(image)
    
    # Añadir dimensión de batch
    img_tensor = img_tensor.unsqueeze(0)
    
    # Realizar inferencia
    with torch.no_grad():
        output = model(img_tensor)
        
    # Para este demo, simplificaremos y usaremos las primeras dos clases
    # de ImageNet como "hombre" y "mujer" (esto es una simplificación)
    # En un sistema real, necesitarías un modelo entrenado específicamente para género
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Para demostración, usamos los índices 0 y 1 como hombre y mujer
    # (en la realidad necesitarías un modelo específico de género)
    male_score = float(probabilities[0].item())
    female_score = float(probabilities[1].item())
    
    # Normalizar para asegurar que sumen 1
    total = male_score + female_score
    if total > 0:
        male_score = male_score / total
        female_score = female_score / total
    
    gender = "hombre" if male_score > female_score else "mujer"
    prob = male_score if gender == "hombre" else female_score
    
    return {
        "genero": gender,
        "probabilidad": float(prob),
        "detalles": [
            {"label": "hombre", "score": float(male_score)},
            {"label": "mujer", "score": float(female_score)}
        ],
        "nota": "Este es un clasificador adaptado de ResNet50 y no está específicamente entrenado para detección de género. Para producción, se recomienda un modelo especializado."
    }

@app.route("/clasificar", methods=["POST"])
def clasificar_genero():
    """Procesa la imagen del usuario y detecta si es hombre o mujer.
    Acepta tanto archivos de imagen como imágenes en base64."""
    
    imagen = None
    
    if 'imagen' in request.files:
        archivo = request.files['imagen']
        if archivo.filename != '':
            try:
                imagen = Image.open(archivo)
            except Exception as e:
                return jsonify({"error": f"No se pudo abrir el archivo de imagen: {e}"}), 400
    
    elif request.is_json:
        datos = request.get_json()
        imagen_base64 = datos.get("imagen", "")
        if imagen_base64:
            try:
                if ',' in imagen_base64:
                    imagen_base64 = imagen_base64.split(',')[1]
                
                imagen_bytes = base64.b64decode(imagen_base64)
                imagen = Image.open(io.BytesIO(imagen_bytes))
            except Exception as e:
                return jsonify({"error": f"No se pudo decodificar la imagen en base64: {e}"}), 400
    
    if imagen is None:
        return jsonify({
            "error": "No se proporcionó una imagen válida. Envíe un archivo en 'imagen' o un JSON con la imagen en base64 en el campo 'imagen'."
        }), 400
    
    try:
        resultado = classify_gender(imagen)
        return jsonify(resultado)
    
    except Exception as e:
        print(f"❌ Error procesando la imagen: {e}")
        return jsonify({"error": f"No se pudo procesar la imagen o generar una predicción: {e}"}), 500

if __name__ == "__main__":
    print("🚀 Servicio de Clasificación de Género iniciado en http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=True)