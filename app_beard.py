import cv2
import numpy as np
from swapper import swap_specific
from face_alignment import FaceAlignment, LandmarksType

# Inicializa o detector de landmarks
fa = FaceAlignment(LandmarksType._2D, flip_input=False)

def extract_landmarks(img):
    lms = fa.get_landmarks(img)
    if lms is None:
        raise ValueError("Face não detectada")
    return lms[0]  # retorna array 68x2

def align_face(src, lms_src, lms_tgt, size):
    pts_src = np.float32([lms_src[36], lms_src[45], lms_src[30]])  # olhos e nariz
    pts_tgt = np.float32([lms_tgt[36], lms_tgt[45], lms_tgt[30]])
    M = cv2.getAffineTransform(pts_src, pts_tgt)
    return cv2.warpAffine(src, M, (size[1], size[0])), M

def get_mask(lms, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    parts = {"L-Eye":range(36,42), "R-Eye":range(42,48), "Nose":range(27,36)}
    for part in parts.values():
        pts = np.array([lms[i] for i in part], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return cv2.GaussianBlur(mask, (15,15), 0)

def main(src_path, tgt_path, out_path):
    src = cv2.imread(src_path)
    tgt = cv2.imread(tgt_path)
    h, w = tgt.shape[:2]

    lms_src = extract_landmarks(src)
    lms_tgt = extract_landmarks(tgt)

    src_aligned, M = align_face(src, lms_src, lms_tgt, (h, w))
    lms_src_aligned = cv2.transform(np.array([lms_src]), M)[0]

    mask = get_mask(lms_src_aligned, h, w)
    swapped = swap_specific(
        source=src_aligned,
        target=tgt,
        landmarks_src=lms_src_aligned,
        landmarks_tgt=lms_tgt,
        mask=mask
    )

    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (tgt * (1 - mask3) + swapped * mask3).astype(np.uint8)
    cv2.imwrite(out_path, result)
    print("Resultado salvo em", out_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Uso: python swap_region.py source.jpg target.jpg output.jpg")
        sys.exit(1)
    main(*sys.argv[1:])
