import cv2
import numpy as np

def classificar_forma(contorno, approx):
    vertices = len(approx)
    
    if vertices == 3:
        return "Triangulo"
        
    elif vertices == 6:
        return "Hexagono"
        
    elif vertices == 10:
        area_contorno = cv2.contourArea(contorno)
        hull = cv2.convexHull(contorno)
        area_hull = cv2.contourArea(hull)
        
        if area_hull > 0:
            solidity = float(area_contorno) / area_hull
            # Aumentei o limite superior da solidez para 0.7 porque 
            # as estrelas da sua imagem são mais "gordinhas"
            if 0.3 < solidity < 0.7:
                return "Estrela"
                
    return None

def processar_imagem(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    if img is None:
        print("Erro ao carregar a imagem.")
        return

    # 1. Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. O Segredo: Binarização Invertida
    # Pixels muito escuros (< 100) viram 255 (Branco). O resto vira 0 (Preto).
    # Isso faz as linhas pretas e os números virarem objetos brancos no escuro.
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 3. Extração de Contornos
    # O RETR_EXTERNAL agora vai contornar o lado de fora das formas brancas.
    # Como as formas são fechadas, ele nunca vai "entrar" nelas para ver os números!
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        # Ignora ruídos minúsculos
        if cv2.contourArea(contorno) < 500:
            continue

        perimetro = cv2.arcLength(contorno, True)
        
        # 4. Ramer-Douglas-Peucker
        # 3% (0.03) do perímetro é uma aproximação excelente para imagens geradas
        epsilon = 0.03 * perimetro
        approx = cv2.approxPolyDP(contorno, epsilon, True)

        nome_forma = classificar_forma(contorno, approx)

        if nome_forma is not None:
            # Desenha o contorno aprovado em verde
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
            
            # Encontra o Centroide
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            # Anotações visuais na imagem original
            cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
            
            # Escreve o nome da forma um pouco acima e à esquerda do centro
            cv2.putText(img, nome_forma, (cX - 45, cY - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostra os resultados
    cv2.imshow("Processamento (Thresh)", thresh) # Mostra a máscara para você entender o truque
    cv2.imshow("Resultado Final", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    processar_imagem("teste.png")