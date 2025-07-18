from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from swapper import process
from PIL import Image
from datetime import datetime 
import io
import os
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)
 
@app.route("/info", methods=['GET'])
def root():
    return f"<h1>Nelltek hair style swap API.All Rights Reserved</h1>"


def upscale_and_sharpen(face_image, scale=2):
    h, w = face_image.shape[:2]
    # Upscale com INTER_LANCZOS4 (melhor qualidade)
    upscaled = cv2.resize(face_image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # Sharpen com kernel simples
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)

    # Redimensiona de volta pro tamanho original pra não zoar o resultado
    final = cv2.resize(sharpened, (w, h), interpolation=cv2.INTER_AREA)
    return final

def color_transfer(source, target):
    # Converte para LAB
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Calcula média e desvio padrão para cada canal
    s_mean, s_std = cv2.meanStdDev(source)
    t_mean, t_std = cv2.meanStdDev(target)

    # Transforma para 1D
    s_mean = s_mean.flatten()
    s_std = s_std.flatten()
    t_mean = t_mean.flatten()
    t_std = t_std.flatten()

    # Aplica a transformação canal a canal
    result = np.zeros_like(target)
    for i in range(3):  # L, A, B
        result[:, :, i] = (target[:, :, i] - t_mean[i]) * (s_std[i] / (t_std[i] + 1e-6)) + s_mean[i]

    # Clipa os valores para faixa válida e converte de volta para uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

@app.route('/faceswap', methods=['POST'])
def faceswap():
    try:

        DATA_FOLDER = '/home/nelljr/nell_hair_style_swap_api/data'
        MODEL_PATH = '/home/nelljr/nell_hair_style_swap_api/checkpoints/inswapper_128.onnx'

        os.makedirs(DATA_FOLDER, exist_ok=True)

        print("🔵 Iniciando processamento faceswap...")
        
        # Receber imagem source como arquivo e target_path como string
        print("Lendo dados do request...")
        source_file = request.files['source']
        target_path = request.form['target_path']
        target_path = target_path.replace("./frontend", "/home/nelljr/nell_hair_style_swap_api/backend")
        
        print(f"Source file: {type(source_file)}")
        print(f"Target path: {target_path}")
        
        # Verificar se o arquivo target existe
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Arquivo target não encontrado: {target_path}")
                
        # Converter source file para PIL Image
        print("Convertendo source para PIL Image...")
        source_img = Image.open(source_file.stream)
        print(f"Source image size: {source_img.size}")
        
        # Carregar target image do sistema de arquivos
        print(f"Carregando target image de: {target_path}")
        target_img = Image.open(target_path)
        print(f"Target image size: {target_img.size}")
        
        
        # Processar face swap
        print("Iniciando processo de face swap...")
        source_img_list = [source_img]  # O inswapper espera uma lista
        result_image = process(source_img_list, target_img, 0, 0, MODEL_PATH)
        print("✅ Face swap concluído!")

        # 🖌️ Aplicar color_transfer para preservar a cor da pele original
        print("🎨 Aplicando color_transfer...")

        # Converter imagens PIL -> OpenCV
        result_cv = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        source_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

        # Redimensionar source para o tamanho do resultado
        source_cv_resized = cv2.resize(source_cv, (result_cv.shape[1], result_cv.shape[0]))

        # Aplicar transferência de cor
        harmonizado_cv = color_transfer(source_cv_resized, result_cv)

        # 🔥 Upscale + Sharpen após color_transfer
        print("🔍 Aplicando upscale + sharpen na imagem final...")
        harmonizado_cv = upscale_and_sharpen(harmonizado_cv)
        print("✅ Upscale + sharpen concluído!")
        
        # Converter de volta para PIL
        result_image = Image.fromarray(cv2.cvtColor(harmonizado_cv, cv2.COLOR_BGR2RGB))

        print("🎉 color_transfer aplicado com suceso!")             

        # Salvar a imagem resultado na pasta ./data
        print("💾 Salvando imagem resultado...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"faceswap_result_{timestamp}.png"
        result_path = os.path.join(DATA_FOLDER, result_filename)
        result_image.save(result_path)
        print(f"✅ Imagem salva em: {result_path}")
        
        # Retornar como arquivo PNG
        print("📤 Preparando retorno para o cliente...")
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        print(f"Buffer size: {len(img_buffer.getvalue())} bytes")
        
        print("✅ Enviando resposta...")
        response = send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=False,
            download_name='faceswap_result.png'
        )
        print("✅ Resposta enviada!")
        return response
            
    except FileNotFoundError as e:
        print(f"❌ Arquivo não encontrado: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Arquivo target não encontrado',
            'message': str(e)
        }), 404
        
    except ValueError as e:
        print(f"❌ Erro de validação: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Erro de validação',
            'message': str(e)
        }), 400
        
    except Exception as e:
        print(f"❌ Erro no processamento: {str(e)}")
        print(f"❌ Tipo do erro: {type(e)}")
        import traceback
        print(f"❌ Stack trace: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Erro ao processar face swap'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
