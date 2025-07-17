import cv2
import dlib
import numpy as np

def get_landmarks(image, detector, predictor):
    """Detecta faces e retorna os pontos de referência faciais."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return None
    # Assume a primeira face detectada
    landmarks = predictor(gray, faces[0])
    points = []
    for i in range(0, 68):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    return np.array(points)

def create_mask(image, landmarks, feature_points):
    """Cria uma máscara para as características faciais especificadas."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Exemplo: pontos para o olho esquerdo e direito e nariz
    # Estes são índices típicos de dlib, você pode precisar ajustar
    # para a sua implementação exata ou para cobrir a área desejada.
    all_points = []
    for feature in feature_points:
        all_points.extend(feature)

    # Cria um contorno convexo dos pontos para cobrir a área
    hull = cv2.convexHull(landmarks[all_points])
    cv2.fillConvexPoly(mask, hull, 255)

    # Você pode querer suavizar a máscara para um blend mais natural
    mask = cv2.GaussianBlur(mask, (7, 7), 0) # Ajuste o kernel conforme necessário
    return mask

def transfer_features(source_image, target_image, detector, predictor):
    source_landmarks = get_landmarks(source_image, detector, predictor)
    target_landmarks = get_landmarks(target_image, detector, predictor)

    if source_landmarks is None or target_landmarks is None:
        print("Não foi possível detectar faces em uma ou ambas as imagens.")
        return None

    # Pontos de referência para olhos e nariz (índices dlib típicos)
    # L-Eye: 36-41
    # R-Eye: 42-47
    # Nose: 27-35
    left_eye_points = list(range(36, 42))
    right_eye_points = list(range(42, 48))
    nose_points = list(range(27, 36)) # Ajustei para incluir a ponta do nariz

    features_to_swap = [left_eye_points, right_eye_points, nose_points]

    # Crie uma cópia da imagem de destino para modificação
    output_image = target_image.copy()

    # Itere sobre as características a serem trocadas
    for feature_indices in features_to_swap:
        # Extraia a característica da imagem de origem
        src_feature_mask = create_mask(source_image, source_landmarks, [feature_indices])
        src_feature_region = cv2.bitwise_and(source_image, source_image, mask=src_feature_mask)

        # Calcule a transformação para alinhar a característica da origem ao destino
        # Isso é a parte mais complexa e crucial. Uma abordagem simplificada seria:
        # 1. Encontrar o centro da característica em ambas as imagens.
        # 2. Calcular a diferença de escala (se necessário, com base em outros pontos de referência).
        # 3. Calcular a rotação (se necessário, com base na orientação da característica).
        # Para um alinhamento mais robusto, você usaria Affine Transformations ou Procrustes Analysis.

        # Para simplificar aqui, vamos tentar um "seamless cloning" com um ponto central.
        # Isso pode não ser perfeito para cada característica individual sem um alinhamento cuidadoso.
        
        # Encontre o centro da característica na imagem de origem e destino
        src_center = tuple(np.mean(source_landmarks[feature_indices], axis=0).astype(int))
        target_center = tuple(np.mean(target_landmarks[feature_indices], axis=0).astype(int))

        # Realize o seamless cloning
        # O seamlessClone funciona melhor quando a máscara e a região de origem estão bem alinhadas.
        # Para características separadas, o alinhamento precisa ser mais preciso.
        try:
            output_image = cv2.seamlessClone(
                src_feature_region,
                output_image,
                src_feature_mask,
                target_center,
                cv2.NORMAL_CLONE # Ou cv2.MIXED_CLONE para resultados diferentes
            )
        except Exception as e:
            print(f"Erro ao aplicar seamlessClone para a característica: {e}")
            print("Verifique se as máscaras e pontos estão corretos.")
            # Se o seamlessClone falhar, você pode tentar um simples blend ou debug

    return output_image

# --- Configuração do dlib ---
# Baixe o modelo de preditor de marcos faciais:
# shape_predictor_68_face_landmarks.dat
# Você pode encontrar isso aqui: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Descompacte e coloque na mesma pasta do seu script ou forneça o caminho completo.
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Carregue suas imagens
    source_img_path = "./david.png"
    target_img_path = "./chris.jpg"

    try:
        source_image = cv2.imread(source_img_path)
        target_image = cv2.imread(target_img_path)

        if source_image is None:
            print(f"Erro: Não foi possível carregar a imagem fonte em {source_img_path}")
        if target_image is None:
            print(f"Erro: Não foi possível carregar a imagem destino em {target_img_path}")

        if source_image is not None and target_image is not None:
            result_image = transfer_features(source_image, target_image, detector, predictor)

            if result_image is not None:
                cv2.imshow("Original Source", source_image)
                cv2.imshow("Original Target", target_image)
                cv2.imshow("Result with Swapped Features", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite("resultado_olhos_nariz.jpg", result_image)
            else:
                print("Não foi possível gerar a imagem resultante.")
        else:
            print("Verifique os caminhos das imagens e se elas existem.")

    except FileNotFoundError:
        print("Erro: Verifique se o arquivo do preditor de marcos faciais existe no caminho especificado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")