from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from swapper import process
from PIL import Image, ImageDraw
from backgroundremover.bg import remove
import io
import os
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)
 
@app.route("/info", methods=['GET'])
def root():
    return f"<h1>Nelltek hair style swap API.All Rights Reserved</h1>"


def upscale_and_sharpen(face_image, scale=3):
    h, w = face_image.shape[:2]
    # Upscale com INTER_LANCZOS4 (melhor qualidade)
    upscaled = cv2.resize(face_image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # Sharpen com kernel simples
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)

    # final = cv2.resize(sharpened, (w, h), interpolation=cv2.INTER_AREA)
    # return final
    return sharpened

    
def remove_background_and_fill_color(pil_image, out_img_path, bg_color):
    # Converte PIL.Image para bytes
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()

    # Remove o fundo com o backgroundremover
    result = remove(
        img_bytes,
        model_name="u2netp",
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_structure_size=10,
        alpha_matting_base_size=1000
    )

    # Carrega a imagem resultante (fundo transparente)
    img_no_bg = Image.open(io.BytesIO(result)).convert("RGBA")

    # Cria novo fundo colorido
    bg = Image.new("RGBA", img_no_bg.size, bg_color + (255,))
    composited = Image.alpha_composite(bg, img_no_bg)

    # Salva resultado final (sem transparência)
    composited.convert("RGB").save(out_img_path)

    
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

        target_path = request.form['target_path']
        
        source_file = request.files['source']
        source_img = Image.open(source_file.stream)
        
        if "beard" in target_path:

            input_path = source_img
            output_path = './data/image_result.png'
            bg_color = (172, 129, 72)

            remove_background_and_fill_color(input_path, output_path, bg_color)

            result_image = Image.open(output_path).convert("RGBA")
            width, height = result_image.size
            center_x, center_y = width // 2, height // 2
            radius = int(min(width, height) * 0.47)  

            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=255
            )

            circle_crop = Image.new("RGBA", (width, height), bg_color + (255,))
            circle_crop.paste(result_image, (0, 0), mask=mask)

            img_buffer = io.BytesIO()
            circle_crop.convert("RGB").save(img_buffer, format='PNG')
            img_buffer.seek(0)

        else:                 

            DATA_FOLDER = '/home/nelljr/nell_hair_style_swap_api/data'
            MODEL_PATH = '/home/nelljr/nell_hair_style_swap_api/checkpoints/inswapper_128.onnx'

            os.makedirs(DATA_FOLDER, exist_ok=True)

            target_path = target_path.replace("./frontend", "/home/nelljr/nell_hair_style_swap_api/backend")           
            target_img = Image.open(target_path)

            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Arquivo target não encontrado: {target_path}")

            source_img_list = [source_img]  # O inswapper espera uma lista
            result_image = process(source_img_list, target_img, 0, 0, MODEL_PATH)

            # Converter imagens PIL -> OpenCV
            result_cv = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            source_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            result_cv_up = upscale_and_sharpen(result_cv)
            source_cv_resized = cv2.resize(source_cv, (result_cv_up.shape[1], result_cv_up.shape[0]))
            # harmonizado_cv = color_transfer(source_cv_resized, result_cv_up)
            harmonizado_cv = source_cv_resized
            
            result_image = Image.fromarray(cv2.cvtColor(harmonizado_cv, cv2.COLOR_BGR2RGB))
            
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
            download_name='image_result.png'
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
