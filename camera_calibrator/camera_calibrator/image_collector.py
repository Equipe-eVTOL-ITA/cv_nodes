import cv2
import os
import time
import sys

# ==========================================
# CONFIGURAÇÕES
# ==========================================
CHECKERBOARD = (7, 6)  # Mesma configuração do camera_calibrator.py
OUTPUT_DIR = 'fotos_calibracao'
MAX_FOTOS = 20

# Sugestões de posições para o Método de Zhang
GUIA_POSICOES = [
    "Centro: Tabuleiro bem de frente, preenchendo o meio.",
    "Longe: Afaste o tabuleiro para pegar o campo de visão total.",
    "Perto: Aproxime o tabuleiro (até quase sair da tela).",
    "Canto Superior Esquerdo: Incline levemente para o centro.",
    "Canto Superior Direito: Incline levemente para o centro.",
    "Canto Inferior Esquerdo: Incline levemente para o centro.",
    "Canto Inferior Direito: Incline levemente para o centro.",
    "Inclinado: Lado esquerdo mais perto da camera.",
    "Inclinado: Lado direito mais perto da camera.",
    "Inclinado: Parte superior mais perto da camera.",
    "Inclinado: Parte inferior mais perto da camera.",
    "Diagonal Principal: Tabuleiro girado 45 graus.",
    "Diagonal Secundaria: Tabuleiro girado -45 graus.",
    "Borda Esquerda: Metade do tabuleiro na borda.",
    "Borda Direita: Metade do tabuleiro na borda.",
    "Borda Superior: Metade do tabuleiro na borda.",
    "Borda Inferior: Metade do tabuleiro na borda.",
    "Distante e Inclinado: Profundidade maxima.",
    "Proximo e Inclinado: Quase tocando na lente.",
    "Final: Qualquer posicao faltante/variada!"
]

def main():
    # Cria diretório de saída
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Diretório '{OUTPUT_DIR}' criado.")

    # Inicializa a câmera (2 é a sua câmera USB identificada)
    camera_index = 2
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Aviso: '{sys.argv[1]}' não é um índice válido. Usando 0.")

    cap = cv2.VideoCapture(camera_index)
    
    # FORCE A RESOLUÇÃO NATIVA 16:9 AQUI
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # (Opcional, mas muito recomendado): Desligue o Autofoco! 
    # O autofoco altera a lente fisicamente e destrói o f_x e f_y durante a captura.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a câmera no índice {camera_index}.")
        return

    print("--- COLETOR DE IMAGENS PARA CALIBRAÇÃO ---")
    print("Comandos:")
    print("  'g' - Tirar foto")
    print("  'q' - Sair")
    
    contador = 0
    # Conta fotos já existentes para não sobrescrever
    files = os.listdir(OUTPUT_DIR)
    contador = len([f for f in files if f.endswith('.jpg')])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao ler frame da câmera.")
            break

        debug_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta o tabuleiro para dar feedback visual
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        # UI: Cor de feedback (Verde se detectado, Vermelho se não)
        cor_status = (0, 255, 0) if found else (0, 0, 255)
        status_txt = "TABULEIRO DETECTADO" if found else "NAO DETECTADO"
        
        # UI: Guia de Posicionamento
        idx_guia = min(contador, len(GUIA_POSICOES) - 1)
        instrucao = GUIA_POSICOES[idx_guia]

        # Desenha na tela (Debug)
        if found:
            cv2.drawChessboardCorners(debug_frame, CHECKERBOARD, corners, found)
        
        # Overlay de Texto
        cv2.putText(debug_frame, f"Fotos: {contador}/{MAX_FOTOS}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, status_txt, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)
        
        # Background para a instrução (melhorar leitura)
        cv2.rectangle(debug_frame, (0, frame.shape[0]-40), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        cv2.putText(debug_frame, f"DICA: {instrucao}", (10, frame.shape[0]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('Coletor de Calibracao - Aperte G para salvar', debug_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('g'):
            if found:
                nome_arquivo = os.path.join(OUTPUT_DIR, f"calib_{contador:02d}.jpg")
                cv2.imwrite(nome_arquivo, frame)
                print(f"Foto salva: {nome_arquivo}")
                contador += 1
                
                # Feedback visual de captura (pisca branco)
                cv2.imshow('Coletor de Calibracao - Aperte G para salvar', 255 * (frame.dtype == 'uint8'))
                cv2.waitKey(50)
            else:
                print("Aviso: Tabuleiro não detectado! Foto não salva.")

        elif key == ord('q'):
            break

        if contador >= MAX_FOTOS:
            print(f"\nAlcançado o objetivo de {MAX_FOTOS} fotos!")
            # Não fecha automaticamente para permitir tirar mais se quiser

    cap.release()
    cv2.destroyAllWindows()
    print("\nProcesso finalizado. Agora você pode rodar 'python3 camera_calibrator.py'")

if __name__ == "__main__":
    main()
