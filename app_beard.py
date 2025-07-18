import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import os

# ================= CONFIGURAÇÕES =================
model_swapper_path = '/home/nelljr/nell_hair_style_swap_api/checkpoints/inswapper_128.onnx'
source_image_path = './david.png'
target_image_path = './chris.png'
output_path = './resultado_faceswap.jpg'

# Verificar arquivos
if not all(map(os.path.exists, [model_swapper_path, source_image_path, target_image_path])):
    raise FileNotFoundError("Um ou mais arquivos não encontrados")

# ================= INICIALIZAÇÃO =================
swapper = ort.InferenceSession(model_swapper_path, providers=['CPUExecutionProvider'])
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

# ================= CARREGAR IMAGENS =================
img_source = cv2.imread(source_image_path)
img_target = cv2.imread(target_image_path)

if img_source is None or img_target is None:
    raise ValueError("Erro no carregamento das imagens")

# ================= DETECÇÃO DE ROSTOS =================
faces_source = app.get(img_source)
faces_target = app.get(img_target)

if not faces_source or not faces_target:
    # Tentativa com limiar mais baixo
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)
    faces_source = app.get(img_source)
    faces_target = app.get(img_target)
    if not faces_source or not faces_target:
        raise ValueError("Detecção de rostos falhou mesmo com limiar baixo")

face_source = faces_source[0]
face_target = faces_target[0]

# ================= PRÉ-PROCESSAMENTO =================
# 1. Embedding para rosto fonte
source_embedding = np.array(face_source.normed_embedding, dtype=np.float32)[np.newaxis]

# 2. Recorte alinhado para rosto alvo
target_face_img = face_align.norm_crop(img_target, landmark=face_target.kps, image_size=128)

# 3. Criar blob (dimensões: 1x3x128x128)
blob = cv2.dnn.blobFromImage(
    target_face_img, 
    1.0 / 255.0, 
    (128, 128), 
    (0, 0, 0), 
    swapRB=True
)

# ================= INFERÊNCIA =================
result = swapper.run(None, {'source': source_embedding, 'target': blob})[0]

# ================= PÓS-PROCESSAMENTO CORRIGIDO =================
# 1. Processar a saída (shape: 3x128x128)
swapped_face = np.clip(result[0] * 255, 0, 255).astype(np.uint8)

# 2. Transpor para formato OpenCV (128x128x3)
swapped_face = swapped_face.transpose(1, 2, 0)

# 3. Converter de RGB para BGR
swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)

# 4. Salvar resultado
cv2.imwrite(output_path, swapped_face)
print(f"Resultado salvo com sucesso em {output_path}")
