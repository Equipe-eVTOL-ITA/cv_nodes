import cv2
import numpy as np
import easyocr

print("Carregando modelo EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True)

def detectar_com_easyocr(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    if img is None:
        print("Erro: Não foi possível carregar a imagem.")
        return

    img_resultado = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        if cv2.contourArea(contorno) < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contorno)

        margem_x = int(w * 0.20)
        margem_y = int(h * 0.20)

        x_inner = x + margem_x
        y_inner = y + margem_y
        w_inner = w - 2 * margem_x
        h_inner = h - 2 * margem_y

        if w_inner <= 0 or h_inner <= 0:
            continue

        roi = img[y_inner:y_inner+h_inner, x_inner:x_inner+w_inner]

        resultados = reader.readtext(roi, allowlist='0123456789')

        for (bbox, texto, confianca) in resultados:
            if confianca > 0.50:
                box_global = np.array(bbox).astype(np.int32) + [x_inner, y_inner]

                # Desenha as caixas e textos
                cv2.polylines(img_resultado, [box_global], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.rectangle(img_resultado, (x, y), (x+w, y+h), (255, 0, 0), 1)
                
                x_texto = box_global[0][0]
                y_texto = box_global[0][1]
                cv2.putText(img_resultado, f"Num: {texto}", (x_texto, y_texto - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                print(f"Sucesso -> Detectado: {texto} | Confiança: {confianca:.2f}")


    nome_arquivo_saida = "resultado_easyocr.png"
    sucesso_salvamento = cv2.imwrite(nome_arquivo_saida, img_resultado)
    
    if sucesso_salvamento:
        print(f"\n[OK] Imagem anotada salva com sucesso: {nome_arquivo_saida}")
    else:
        print("\n[ERRO] Falha ao salvar a imagem no disco.")

if __name__ == "__main__":
    detectar_com_easyocr("teste.png")