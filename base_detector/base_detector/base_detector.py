import cv2 as cv
import math
if __name__ == '__main__':
    frame = cv.imread("base.png")
    if frame is None:
        print("Error: base.png not found")
        exit()

    while True:
        # We don't need camera.read() anymore, using the static frame
        # Make a copy for drawing so we don't overwrite the original in each loop iteration
        frame_vis = frame.copy()
        
        cinza = cv.cvtColor(frame_vis, cv.COLOR_BGR2GRAY)

        blured = cv.GaussianBlur(cinza, (7, 7), 0)

        sobel_x = cv.Sobel(blured, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(blured, cv.CV_64F, 0, 1, ksize=3)

        sobel_x_abs = cv.convertScaleAbs(sobel_x)
        sobel_y_abs = cv.convertScaleAbs(sobel_y)

        sobel_completo = cv.bitwise_or(sobel_x_abs, sobel_y_abs)

        _, binario = cv.threshold(sobel_completo, 50, 255, cv.THRESH_BINARY)

        contornos, _ = cv.findContours(binario, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        circulos_encontrados = []
        for c in contornos:
            area = cv.contourArea(c)

            if area < 100:
                continue # filtra os círculos muito pequenos (noise)

            perimetro = cv.arcLength(c, True)

            if perimetro == 0:
                continue

            circularidade = (4*math.pi*area)/(perimetro*perimetro)
            print(circularidade)

            if 0.5 < circularidade < 1.2:
                circulos_encontrados.append(c)
                
                # Image moments to find the centroid
                M = cv.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw a small circle at the centroid
                    cv.circle(frame_vis, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Display coordinates
                    cv.putText(frame_vis, f"C: ({cx}, {cy})", (cx-20, cy + 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv.drawContours(frame_vis, circulos_encontrados, -1, (0,255,0), thickness=2)

        cv.imshow('deteccao', frame_vis)
        cv.imshow('binario', binario)

        if cv.waitKey(20)&0xFF == ord('d'):
            break

    cv.destroyAllWindows()
