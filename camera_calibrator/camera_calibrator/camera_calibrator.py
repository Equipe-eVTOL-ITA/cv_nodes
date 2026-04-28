import numpy as np
import cv2
import glob

# ==========================================
# PARÂMETROS DO TABULEIRO
# ==========================================
# Número de INTERSEÇÕES internas (colunas, linhas)
# Ex: Um tabuleiro de 9x7 quadrados tem 8x6 interseções.
CHECKERBOARD = (7, 6)

# Tamanho exato do lado de um quadrado impresso (em metros)
# Use um paquímetro para medir. Ex: 25mm = 0.025
TAMANHO_QUADRADO_M = 0.032 

# Critério de parada para a otimização subpixel (precisão máxima)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepara os pontos 3D teóricos do tabuleiro no mundo real
# Formato: (0,0,0), (1,0,0), (2,0,0) ... multiplicado pelo tamanho em metros
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * TAMANHO_QUADRADO_M

# Arrays para armazenar os pontos 3D e 2D de todas as imagens
objpoints = [] # Pontos 3D no mundo real
imgpoints = [] # Pontos 2D no plano da imagem

# Carrega as imagens (altere a extensão se usar .png)
images = glob.glob('fotos_calibracao/*.jpg')

if not images:
    print("Nenhuma imagem encontrada na pasta!")
    exit()

print(f"Processando {len(images)} imagens...")

shape_imagem = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if shape_imagem is None:
        shape_imagem = gray.shape[::-1] # (largura, altura)

    # Encontra as quinas do tabuleiro na imagem
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        objpoints.append(objp)
        
        # Refina a coordenada dos pixels encontrados para precisão subpixel
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # (Opcional) Desenha e mostra as quinas para debug visual
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Verificacao das Quinas', img)
        cv2.waitKey(100) # Mostra por 100ms

cv2.destroyAllWindows()

# ==========================================
# CÁLCULO DA CALIBRAÇÃO
# ==========================================
print("\nCalculando matriz intrinseca... Isso pode levar alguns segundos.")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape_imagem, None, None)

print("\n=== RESULTADOS DA CALIBRAÇÃO ===")
print("Erro de Reprojeção RMS (quanto mais próximo de 0, melhor):")
print(ret)

print("\nMatriz Intrínseca (K):")
print(mtx)

print("\nCoeficientes de Distorção (k1, k2, p1, p2, k3):")
print(dist)

# Salva os arrays para você carregar no seu nó do ROS2 depois
np.savez('camera_params.npz', mtx=mtx, dist=dist)
print("\nParâmetros salvos em 'camera_params.npz'")