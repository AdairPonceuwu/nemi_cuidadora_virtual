import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # evita conflictos OpenMP

import cv2, argparse, json, time, pandas as pd, numpy as np, unicodedata, re
from datetime import timedelta
from rapidocr_onnxruntime import RapidOCR
from collections import deque

# Notificaciones (opcional)
try:
    from win10toast import ToastNotifier
except Exception:
    ToastNotifier = None

# Fuzzy matching (rapidfuzz si está disponible; fallback a difflib)
try:
    from rapidfuzz import fuzz
    def fuzzy_ratio(a, b):
        return max(fuzz.partial_ratio(a, b), fuzz.token_set_ratio(a, b)) / 100.0
except Exception:
    from difflib import SequenceMatcher
    def fuzzy_ratio(a, b):
        return SequenceMatcher(None, a, b).ratio()

LEET = str.maketrans({"0":"o","1":"i","3":"e","4":"a","5":"s","7":"t","@":"a","$":"s","!":"i"})

def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.translate(LEET)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_forbidden(list_str: str, file_path: str | None):
    phrases = []
    if list_str:
        phrases += [norm_text(p) for p in list_str.split(",") if p.strip()]
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            phrases += [norm_text(line) for line in f if line.strip()]
    if not phrases:
        phrases = [
            "te deseo","te quiero","mandame fotos","ven a mi casa",
            "pack","sexo","nudes","pasa tu numero","dm por privado", "ganas de ti"
        ]
    return sorted({p for p in phrases if p})

def sec_to_hhmmss(s): return str(timedelta(seconds=round(s)))

def adjust_gamma(img_bgr, gamma=1.0):
    if gamma is None or abs(gamma-1.0) < 1e-3:
        return img_bgr
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([((i/255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)

def upscale(img_bgr, scale=1.0):
    if scale is None or scale <= 1.01:
        return img_bgr
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def preprocess_roi(img_bgr, mode="light", scale=1.0, gamma=1.0, invert=False):
    """Preprocesado robusto para texto de UI translúcida."""
    img = upscale(img_bgr, scale=scale)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    if mode == "none":
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return adjust_gamma(out, gamma)

    if mode == "light":
        g = cv2.GaussianBlur(g, (3, 3), 0)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return adjust_gamma(out, gamma)

    # strong
    g = cv2.bilateralFilter(g, 5, 50, 50)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 11)
    if invert:
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)
    out = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return adjust_gamma(out, gamma)

def parse_roi_arg(s):
    try:
        x, y, w, h = [int(v) for v in s.split(",")]
        if w <= 0 or h <= 0: raise ValueError
        return {"x": x, "y": y, "w": w, "h": h}
    except Exception:
        raise argparse.ArgumentTypeError("ROI debe ser 'x,y,w,h' con enteros positivos.")

def main():
    ap = argparse.ArgumentParser(description="Roblox Chat Guard (local, RapidOCR/ONNX) con detección mejorada")
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=float, default=1.0, help="FPS de análisis")
    ap.add_argument("--roi_json", default="roi_chat.json")
    ap.add_argument("--headless", action="store_true", help="No abrir selector; usar ROI guardado o --roi")
    ap.add_argument("--roi", type=parse_roi_arg, help="ROI manual x,y,w,h")

    # Mejora de OCR
    ap.add_argument("--preprocess", default="light", choices=["none","light","strong"])
    ap.add_argument("--scale", type=float, default=1.5, help="Escala del ROI antes del OCR (1.0–2.0)")
    ap.add_argument("--gamma", type=float, default=1.2, help="Corrección gamma (1.0=sin cambio)")
    ap.add_argument("--invert", action="store_true", help="Invertir blanco/negro en preprocesamiento fuerte")
    ap.add_argument("--passes", type=int, default=2, help="Pasadas de OCR por frame (1=rápido, 2=robusto)")
    ap.add_argument("--min_conf", type=float, default=0.6, help="Confianza mínima RapidOCR (0-1)")

    # Detección
    ap.add_argument("--forbidden", type=str, default="te deseo,sexo,pack,mandame fotos,ven a mi casa, ganas de ti")
    ap.add_argument("--forbidden_file", type=str)
    ap.add_argument("--fuzzy", type=float, default=0.85, help="Umbral [0-1] similitud fuzzy; usa 0 para desactivar")
    ap.add_argument("--history", type=int, default=3, help="Frames a agregar para detección temporal")
    ap.add_argument("--min_print_gap", type=float, default=5.0, help="Segundos entre alertas idénticas")

    # Salidas
    ap.add_argument("--out_csv", default="chat_alertas_local.csv")
    ap.add_argument("--out_video", default="chat_guard_local_annotated.mp4")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log_txt", default="chat_alertas_local.txt")
    ap.add_argument("--toast", action="store_true")
    ap.add_argument("--toast_title", default="Roblox Chat Guard")
    args = ap.parse_args()

    fuzzy_thr = None if args.fuzzy <= 0 else float(args.fuzzy)
    forbidden_norm = parse_forbidden(args.forbidden, args.forbidden_file)
    print("[INFO] Frases prohibidas:", forbidden_norm)

    ocr = RapidOCR()  # sin GPU/Torch

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir: {args.video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ROI
    roi = None
    if args.roi:
        roi = args.roi
    elif os.path.exists(args.roi_json):
        roi = json.load(open(args.roi_json, "r", encoding="utf-8"))
    elif args.headless:
        raise SystemExit("Headless sin ROI. Pasa --roi x,y,w,h o ejecuta sin --headless para seleccionar.")
    else:
        ok, f0 = cap.read()
        if not ok: raise SystemExit("No se pudo leer el primer frame.")
        win = "Selecciona ROI del chat (ENTER)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        tw = min(1280, W); th = int(tw * H / W); cv2.resizeWindow(win, tw, th)
        x,y,w,h = map(int, cv2.selectROI(win, f0, showCrosshair=True, fromCenter=False))
        cv2.destroyAllWindows()
        if w==0 or h==0: raise SystemExit("ROI inválido.")
        roi = {"x":x,"y":y,"w":w,"h":h}
        json.dump(roi, open(args.roi_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    x,y,w,h = roi["x"], roi["y"], roi["w"], roi["h"]

    # salida de video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_video, fourcc, max(1.0, args.fps), (W, H))

    # muestreo
    step = max(int(round(native_fps / args.fps)), 1)
    rows=[]; idx=0
    last_alert_time_by_phrase = {}
    agg_history = deque(maxlen=max(1, args.history))
    os.makedirs("hits", exist_ok=True)

    toaster = ToastNotifier() if (args.toast and ToastNotifier is not None) else None
    if args.toast and toaster is None:
        print("[WARN] win10toast no instalado; sin notificaciones.")

    print("[INFO] Iniciando… Ctrl+C para salir")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % step != 0:
                idx += 1
                continue

            ts = idx / native_fps
            roi_frame = frame[y:y+h, x:x+w].copy()

            # --- MÚLTIPLES PASADAS DE OCR ---
            all_lines = []
            passes = max(1, args.passes)
            for p in range(passes):
                mode = "light" if p == 0 else "strong"
                inv = False if p == 0 else args.invert
                roi_proc = preprocess_roi(roi_frame, mode=mode, scale=args.scale, gamma=args.gamma, invert=inv)

                result, _ = ocr(roi_proc)  # RapidOCR -> [[pts, text, score], ...]
                if result:
                    for pts, text, score in result:
                        if text and float(score) >= args.min_conf:
                            all_lines.append((pts, text.strip(), float(score)))

            # Debug
            if args.debug and all_lines:
                print(f"[{sec_to_hhmmss(ts)}] OCR lines ({len(all_lines)}):")
                for _, t, sc in all_lines[:8]:
                    print(f"  ({sc:.2f}) raw: {t} | norm: {norm_text(t)}")

            # Agregación temporal (últimos N frames)
            agg_texts = [norm_text(t) for _, t, _ in all_lines]
            agg_history.append(" ".join(agg_texts))
            agg_concat = " ".join(agg_history)

            # --- DETECCIÓN ---
            flagged, hit_phrase, hit_line = False, None, None

            # 1) substring en el texto agregado
            for pword in forbidden_norm:
                if pword and pword in agg_concat:
                    flagged, hit_phrase = True, pword
                    # intenta recuperar una línea específica del frame actual
                    hit_line = next((t for _, t, _ in all_lines if pword in norm_text(t)), "match temporal")
                    break

            # 2) fuzzy por línea y por agregado (si aplica)
            if not flagged and fuzzy_thr is not None:
                # líneas actuales
                for _, t, _sc in all_lines:
                    ln = norm_text(t)
                    for pword in forbidden_norm:
                        if pword and fuzzy_ratio(ln, pword) >= fuzzy_thr:
                            flagged, hit_phrase, hit_line = True, pword, t
                            break
                    if flagged: break
                # agregado temporal (un poco más permisivo)
                if not flagged and agg_concat:
                    thr_agg = max(0.9 * fuzzy_thr, 0.75)
                    for pword in forbidden_norm:
                        if pword and fuzzy_ratio(agg_concat, pword) >= thr_agg:
                            flagged, hit_phrase, hit_line = True, pword, "(match temporal)"
                            break

            # --- OVERLAY / DIBUJOS ---
            color = (0,0,255) if flagged else (255,255,255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Dibuja boxes del OCR
            for pts, t, sc in all_lines:
                poly = np.array([(x+int(px), y+int(py)) for px,py in pts], dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(frame, [poly], True, (0,255,0), 1)
                tx, ty = poly[0,0,0], max(poly[0,0,1]-4, 0)
                cv2.putText(frame, f"{t[:38]} ({sc:.0%})", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

            if flagged:
                tnow = time.time()
                prev = last_alert_time_by_phrase.get(hit_phrase, 0)
                if tnow - prev >= args.min_print_gap:
                    print(f"⚠️  ALERTA {sec_to_hhmmss(ts)} | match='{hit_phrase}' | texto='{hit_line}'")
                    last_alert_time_by_phrase[hit_phrase] = tnow
                    # snapshot
                    snap = f"hits/hit_{int(ts)}.png"
                    cv2.imwrite(snap, frame)
                    # beep opcional
                    try:
                        import winsound; winsound.Beep(1200, 180)
                    except Exception:
                        pass
                    # log TXT
                    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        with open(args.log_txt, "a", encoding="utf-8") as f:
                            f.write(f"[{stamp}] {sec_to_hhmmss(ts)} | {hit_phrase} | {hit_line}\n")
                    except Exception as e:
                        print("[WARN] No se pudo escribir TXT:", e)
                    # toast
                    if toaster:
                        try:
                            toaster.show_toast(args.toast_title, f"{hit_phrase}\n{hit_line[:90]}", duration=5, threaded=True)
                        except Exception as e:
                            print("[WARN] Toast error:", e)

                cv2.putText(frame, f"ALERTA: {hit_phrase}", (x, max(15, y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

                rows.append({
                    "timestamp_sec": round(ts,2),
                    "timestamp_hhmmss": sec_to_hhmmss(ts),
                    "frase": hit_phrase,
                    "linea": hit_line
                })

            if args.preview:
                cv2.imshow("Roblox Chat Guard (local)", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            out.write(frame)
            idx += 1

    finally:
        cap.release(); out.release()
        if args.preview:
            cv2.destroyAllWindows()
        pd.DataFrame(rows, columns=["timestamp_sec","timestamp_hhmmss","frase","linea"]).to_csv(
            args.out_csv, index=False, encoding="utf-8-sig")
        print("[OK] CSV:", args.out_csv, "| Video:", args.out_video, "| Hits:", len(rows))

if __name__ == "__main__":
    main()
