import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# 1. Carregar modelo e detectar rostos (mesmo do exemplo anterior)
model_swapper_path = '/home/nelljr/nell_hair_style_swap_api/checkpoints/inswapper_128.onnx'
swapper = ort.InferenceSession(model_swapper_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Carregar imagens
img_source = cv2.imread('./david.png')
img_target = cv2.imread('./chris.png')

faces_source = app.get(img_source)
faces_target = app.get(img_target)

# 2. Gerar rosto trocado completo (128x128)
face_source = faces_source[0]
face_target = faces_target[0]

# Extrair embedding da fonte
source_embedding = np.array(face_source.normed_embedding, dtype=np.float32)[None]

# Preparar rosto alvo (normalizado)
target_face_img = face_target.normed_crop  # Imagem 128x128

# Executar troca completa
blob = cv2.dnn.blobFromImage(target_face_img, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
result = swapper.run(None, {'source': source_embedding, 'target': blob})[0]
swapped_face = np.clip(result[0] * 255, 0, 255).astype(np.uint8)
swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)

# 3. Criar máscaras para olhos e nariz
def create_region_mask(landmarks, region_indices, img_size=128):
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    points = np.array([landmarks[i] for i in region_indices], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

# Índices aproximados dos landmarks (ajuste conforme seu modelo)
LEFT_EYE_INDICES = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42]   # Olho esquerdo
RIGHT_EYE_INDICES = [87, 88, 89, 90, 91, 92, 93, 94, 95, 96]  # Olho direito
NOSE_INDICES = [51, 52, 53, 54, 55, 56, 57, 58, 59]           # Nariz

# Criar máscaras combinadas
landmarks = face_target.landmark.astype(np.int32)
eyes_nose_mask = create_region_mask(landmarks, LEFT_EYE_INDICES + RIGHT_EYE_INDICES + NOSE_INDICES)

# 4. Aplicar apenas olhos e nariz na imagem alvo
# Redimensionar para o tamanho original do rosto na imagem alvo
swapped_regions = cv2.bitwise_and(swapped_face, swapped_face, mask=eyes_nose_mask)
original_face_regions = cv2.bitwise_and(target_face_img, target_face_img, mask=~eyes_nose_mask)

# Combinar regiões
final_face = cv2.add(original_face_regions, swapped_regions)

# 5. Colocar de volta na imagem original (simplificado)
# (Para implementação real, use transformação inversa com os landmarks)
x, y, w, h = face_target.bbox.astype(int)
resized_face = cv2.resize(final_face, (w, h))
img_target[y:y+h, x:x+w] = resized_face

# Salvar resultado
cv2.imwrite('resultado_parcial.jpg', img_target)