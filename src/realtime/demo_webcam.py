"""Protótipo em tempo real do Pipeline A: webcam -> MediaPipe -> EAR/PERCLOS -> alerta.

Uso:
    python src/realtime/demo_webcam.py [--camera 0] [--ear-thresh 0.21]

Lógica de alerta (com amortecimento temporal para ignorar piscadas normais):
  - OLHOS FECHADOS: EAR abaixo do limiar por >= `--closed-s` segundos seguidos.
  - PERCLOS: fração de quadros com olho fechado na última janela de 60 s
    acima de `--perclos-thresh`.
  - BOCEJO: MAR acima de 0.6 por >= 2 s.

Nos primeiros segundos o limiar de EAR é calibrado automaticamente com o
olho aberto do usuário (75% da mediana), a menos que --ear-thresh seja dado.

Teclas: q = sair.
"""

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

from metrics import Perclos, ear_from_landmarks, mar_from_landmarks

CALIB_S = 3.0
MAR_THRESH = 0.6
YAWN_S = 2.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0, help="índice da webcam")
    ap.add_argument("--ear-thresh", type=float, default=None,
                    help="limiar de EAR (padrão: calibração automática)")
    ap.add_argument("--closed-s", type=float, default=1.0,
                    help="segundos de olho fechado para disparar alerta")
    ap.add_argument("--perclos-thresh", type=float, default=0.15,
                    help="limiar de PERCLOS na janela de 60 s")
    return ap.parse_args()


def main():
    args = parse_args()
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a câmera {args.camera}")

    perclos = Perclos(window_s=60.0)
    ear_thresh = args.ear_thresh
    calib_vals = []
    t0 = time.monotonic()
    closed_since = None
    yawn_since = None
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        h, w = frame.shape[:2]
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        alert_msgs = []
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            ear = ear_from_landmarks(lms, w, h)
            mar = mar_from_landmarks(lms, w, h)

            # Calibração automática do limiar com olho aberto
            if ear_thresh is None:
                calib_vals.append(ear)
                cv2.putText(frame, "Calibrando... mantenha os olhos abertos",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if now - t0 >= CALIB_S:
                    ear_thresh = 0.75 * float(np.median(calib_vals))
                    print(f"[calibração] limiar de EAR = {ear_thresh:.3f}")
            else:
                closed = ear < ear_thresh
                p = perclos.update(now, closed)

                if closed:
                    closed_since = closed_since or now
                    if now - closed_since >= args.closed_s:
                        alert_msgs.append("ALERTA: OLHOS FECHADOS")
                else:
                    closed_since = None

                if p > args.perclos_thresh:
                    alert_msgs.append(f"ALERTA: PERCLOS {p:.0%}")

                if mar > MAR_THRESH:
                    yawn_since = yawn_since or now
                    if now - yawn_since >= YAWN_S:
                        alert_msgs.append("AVISO: BOCEJO")
                else:
                    yawn_since = None

                hud = [f"EAR {ear:.3f} (limiar {ear_thresh:.3f})",
                       f"PERCLOS(60s) {p:.1%}", f"MAR {mar:.2f}", f"FPS {fps:.1f}"]
                for i, txt in enumerate(hud):
                    cv2.putText(frame, txt, (10, 30 + 25 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Rosto nao detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        for i, msg in enumerate(alert_msgs):
            cv2.putText(frame, msg, (10, h - 60 - 35 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        if alert_msgs:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            print("\a", end="", flush=True)  # bipe do terminal

        fps_n += 1
        if now - fps_t >= 1.0:
            fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

        cv2.imshow("Deteccao de Sonolencia - Pipeline A (prototipo)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
