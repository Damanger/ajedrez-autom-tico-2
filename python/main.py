import cv2
import mediapipe as mp
import numpy as np
import socket
import time

# Configuración de Red (IPC)
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Puerto no privilegiado (>1023)

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def find_position(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        lm_list = []
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
        return lm_list

def get_square(x, y, width, height):
    """Mapeo de coordenadas cartesianas (pixeles) a algebraicas (ajedrez)."""
    # Invertimos eje X para efecto espejo natural
    x = width - x 
    col = int(x / (width / 8))
    row = int(y / (height / 8))
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    if 0 <= col < 8 and 0 <= row < 8:
        return files[col] + ranks[row]
    return None

def send_move_to_haskell(move_uci):
    """Envía la jugada al servidor Haskell mediante TCP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(move_uci.encode('utf-8'))
        print(f"[IPC] Enviado a Haskell: {move_uci}")
        return True
    except ConnectionRefusedError:
        print("[Error] El servidor Haskell no está escuchando.")
        return False

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    dragging = False
    start_sq = None
    
    print("Sistema de Visión Iniciado. Esperando conexión...")

    while True:
        success, img = cap.read()
        if not success: break
        
        h, w, _ = img.shape
        
        # Renderizado de Grid para referencia visual (Realidad Aumentada simple)
        step_x, step_y = w // 8, h // 8
        for i in range(1, 8):
            cv2.line(img, (i * step_x, 0), (i * step_x, h), (200, 200, 200), 1)
            cv2.line(img, (0, i * step_y), (w, i * step_y), (200, 200, 200), 1)

        lm_list = tracker.find_position(img)
        
        if len(lm_list) != 0:
            # Puntos 4 (Pulgar) y 8 (Índice)
            x4, y4 = lm_list[4][1], lm_list[4][2]
            x8, y8 = lm_list[8][1], lm_list[8][2]
            cx, cy = (x4 + x8) // 2, (y4 + y8) // 2
            
            # Distancia euclidiana
            length = np.hypot(x8 - x4, y8 - y4)
            
            current_sq = get_square(cx, cy, w, h)
            
            # Umbral de activación
            if length < 40: 
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                if not dragging:
                    dragging = True
                    start_sq = current_sq
            else:
                if dragging: # Evento "Soltar" (MouseUp)
                    dragging = False
                    end_sq = current_sq
                    if start_sq and end_sq and start_sq != end_sq:
                        move_uci = f"{start_sq}{end_sq}"
                        # Lógica de promoción simple (siempre dama si es peón en última fila)
                        # Se puede refinar consultando el tablero, pero por ahora enviamos base.
                        send_move_to_haskell(move_uci)
                        time.sleep(0.5) # Debounce para evitar dobles envíos
                    start_sq = None

            if current_sq:
                cv2.putText(img, f"Pos: {current_sq}", (10, h - 20), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        cv2.imshow("Vision Chess Sensor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()