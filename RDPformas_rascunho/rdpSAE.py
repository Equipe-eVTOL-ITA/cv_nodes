import cv2 as cv
import numpy as np
import math

def angulo_valido(approx, limite_min_graus=20, limite_max_graus=150):
    n = len(approx)
    if n < 3: return False
    
    for i in range(n):
        # Pega 3 pontos seguidos (fazendo o wrap com o resto da divisão para o último ponto)
        p1 = approx[i][0]
        p2 = approx[(i + 1) % n][0] # Vértice do ângulo
        p3 = approx[(i + 2) % n][0]
        
        # Cria os vetores u e v
        u = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Produto escalar e normas
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        # Evita divisão por zero
        if norm_u == 0 or norm_v == 0: continue
        
        # Calcula o ângulo em graus
        cos_theta = dot_product / (norm_u * norm_v)
        # Clip para evitar erros de precisão numérica fora de [-1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angulo = math.degrees(math.acos(cos_theta))
        
        if angulo < limite_min_graus or angulo > limite_max_graus:
            return False # Se achar um ângulo "fechado" demais, descarta o contorno
            
    return True

camera = cv.VideoCapture(1)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # ==========================================================
    # 1. PRÉ-PROCESSAMENTO
    # ==========================================================
    img_cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # blur Gaussiano pesado (11x11) para matar o ruído
    gaussblur = cv.GaussianBlur(img_cinza, (11, 11), 0)

    # filtro Scharr (mais sensível e preciso que o Sobel)
    grad_x = cv.Scharr(gaussblur, cv.CV_16S, 1, 0)
    grad_y = cv.Scharr(gaussblur, cv.CV_16S, 0, 1)


    # Otsu para achar o threshold ideal baseado na luz atual
    threshold_alta, _ = cv.threshold(gaussblur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
    threshold_baixa = threshold_alta * 0.9
        
    # Canny recebendo os gradientes e calculando com L2gradient
    bordas = cv.Canny(grad_x, grad_y, threshold_baixa, threshold_alta, L2gradient=True)

    """
    binaria = cv.adaptiveThreshold(gaussblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    bordas = cv.Canny(binaria, 50, 150)
    """

    # cola as quinas e linhas que ficaram falhadas/pontilhadas
    kernel = np.ones((4,4), np.uint8)
    bordasfinal = cv.morphologyEx(bordas, cv.MORPH_CLOSE, kernel)


    # ==========================================================
    # 2. ENCONTRANDO CONTORNOS E FILTRANDO O LIXO
    # ==========================================================
    contours, hierarchy = cv.findContours(bordasfinal, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # contornos_aprovados = []
    
    outputRDP = frame.copy()

    if hierarchy is not None:
        hierarchy = hierarchy[0]

        for i, cnt in enumerate(contours):
            paiIndex = hierarchy[i][3]

            if paiIndex == -1:
                area = cv.contourArea(cnt)
                perimetro = cv.arcLength(cnt, True)
                if perimetro == 0: continue

                circularidade = (4 * math.pi * area) / (perimetro ** 2)

                if circularidade > 0.70 and area > 3700:
                    for j, filho_cnt in enumerate(contours):
                        if hierarchy[j][3] == i:
                            area_filho = cv.contourArea(filho_cnt)
                            if area_filho < 2300: continue
                            perim_filho = cv.arcLength(filho_cnt, True)
                            if perim_filho == 0: continue
                            circularidade_filho = (4 * math.pi * area_filho) / (perim_filho ** 2)
                            epsilon = 0.02 * perim_filho
                            approx = cv.approxPolyDP(filho_cnt, epsilon, True)
                            quantidade_de_pontos = len(approx)

                            hull = cv.convexHull(filho_cnt) # Cria um contorno convexo (hull) para filho_cnt
                            area_hull = cv.contourArea(hull) # Calcula a área de hull
                            perim_hull = cv.arcLength(hull, True) # Calcula o perímetro de hull
                            circularidade_hull = (4 * math.pi * area_hull) / (perim_hull ** 2)

                            solidez = area_filho / float(area_hull) if area_hull > 0 else 0 # Calcula a solidez

                            approx2 = cv.approxPolyDP(hull, epsilon, True)
                            qtd_pontos_hull = len(approx2)

                            if not angulo_valido(approx, 10, 165): continue
                            if not angulo_valido(approx2, 20, 135): continue

                            # Checagem das formas geométricas

                            if quantidade_de_pontos == 4 or qtd_pontos_hull == 4:
                                continue

                            if quantidade_de_pontos == 3 and (0.520 <= circularidade_filho <= 0.690):
                                x, y, w, h = cv.boundingRect(filho_cnt)
                                cv.putText(outputRDP, "Triangulo", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                print(f"Triângulo detectado! Circularidade: {circularidade:.2f}")

                            elif quantidade_de_pontos == 6 and (0.700 <= circularidade_filho <= 0.970):
                                x, y, w, h = cv.boundingRect(filho_cnt)
                                cv.putText(outputRDP, "Hexagono", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                print(f"Hexágono detectado! Circularidade: {circularidade:.2f}")
                            
                            elif qtd_pontos_hull == 5 and (0.775 <= circularidade_hull <= 0.945) and quantidade_de_pontos > 5:
                                x, y, w, h = cv.boundingRect(filho_cnt)
                                cv.putText(outputRDP, "Estrela", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                print(f"Estrela detectada! Circularidade: {circularidade:.2f}")

                            # Desenha os contornos aprovados em verde e os vértices em vermelho
                            cv.drawContours(outputRDP, [approx], -1, (0, 255, 0), 2)
                            for ponto in approx:
                                px, py = ponto[0]
                                cv.circle(outputRDP, (px, py), 4, (0, 0, 255), -1)

    # ==========================================================
    # 4. EXIBIÇÃO NA TELA
    # ==========================================================

    cv.imshow('Visao do Canny (Bordas Final)', bordasfinal)
    cv.imshow('Visao do Drone (Resultado Final)', outputRDP)

    if cv.waitKey(20)&0xFF == ord('d'):
        break

camera.release()
cv.destroyAllWindows()