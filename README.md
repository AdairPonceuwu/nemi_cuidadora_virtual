<img width="1088" height="739" alt="image" src="https://github.com/user-attachments/assets/5bbcbc92-047e-4747-b749-d4b129568b0a" /># Nemi

Detección en vivo de frases peligrosas en el **chat de Roblox** con OCR local (RapidOCR/ONNXRuntime).  
Funciona **offline**, anota el video, genera **alertas**, guarda **CSV**, **snapshots** y (opcional) **notificaciones de Windows**.
<img width="1088" height="739" alt="image" src="https://github.com/user-attachments/assets/e28010a2-7ef2-4b69-962f-f1d25c3274a7" />


---

## ✨ Características
- OCR 100% local (sin claves ni servicios externos).
- Detección robusta con normalización + *fuzzy matching* (tolera errores de OCR como “te de5eo”).
- Preprocesado configurable (escala, gamma, inversión, múltiples pasadas).
- Agregación temporal (combina N frames para no perder frases “cortadas”).
- Salidas: video anotado, CSV de alertas, logs en TXT y capturas (`/hits`).

---

## 📦 Requisitos

- Python 3.9–3.11
- Windows/macOS/Linux

**Instalación rápida (Conda recomendado):**
```bash
conda create -n chat-guardian python=3.10 -y
conda activate chat-guardian
pip install rapidocr-onnxruntime opencv-python pandas numpy rapidfuzz win10toast
```
> `rapidfuzz` y `win10toast` son opcionales (mejor fuzzy / notificaciones Windows).

**Alternativa con requirements.txt**
```txt
rapidocr-onnxruntime
opencv-python
pandas
numpy
rapidfuzz
win10toast
```
```bash
pip install -r requirements.txt
```

---

## 🚀 Uso básico

Guarda el script como **`nemi.py`** y corre:

### 1) Seleccionando ROI con ventana
```bash
python nemi.py --video partida_roblox.mp4 --preview
```
- Aparecerá una ventana: selecciona con el mouse el **cuadro del chat** y presiona **ENTER**.
- Se guardará `roi_chat.json` para reutilizar.

### 2) ROI fijo (sin ventana)
```bash
python nemi.py --video partida_roblox.mp4   --roi 40,60,520,360 --headless --preview
```

### 3) Comando recomendado (ejemplo real)
```bash
python nemi.py --video partida_roblox.mp4   --preview --preprocess light --scale 1.2 --gamma 1.1   --passes 1 --history 2 --fuzzy 0.83
```

---

## ⚙️ Opciones

```
--video PATH                 Ruta del video (obligatoria)
--fps FLOAT                  FPS de análisis (por defecto 1.0)

--roi_json FILE              Archivo con el ROI guardado (por defecto roi_chat.json)
--headless                   No abrir ventana; usar ROI guardado o --roi
--roi x,y,w,h                ROI manual (ej. 40,60,520,360)

# Mejora de OCR
--preprocess [none|light|strong]  Preprocesado del ROI
--scale FLOAT               Escala del ROI antes del OCR (1.0–2.0; ej. 1.2 o 2.0)
--gamma FLOAT               Corrección gamma (ej. 1.1–1.3)
--invert                    Invierte blanco/negro (útil con fondos claros)
--passes INT                Pasadas de OCR por frame (1 rápido, 2 más robusto)
--min_conf FLOAT            Confianza mínima de RapidOCR (0–1, ej. 0.6)

# Detección
--forbidden "a,b,c"         Frases peligrosas separadas por coma
--forbidden_file FILE       Archivo .txt con una frase por línea
--fuzzy FLOAT               Umbral de similitud (0–1). Ej. 0.8–0.85
--history INT               Frames a agregar para detección temporal (ej. 2–3)
--min_print_gap FLOAT       Segundos entre alertas iguales (evita spam)

# Salidas
--out_csv FILE              CSV de alertas (por defecto chat_alertas_local.csv)
--out_video FILE            Video anotado (por defecto chat_guard_local_annotated.mp4)
--log_txt FILE              Log en texto (por defecto chat_alertas_local.txt)
--preview                   Mostrar ventana en vivo
--debug                     Imprimir texto OCR (raw y normalizado) en consola
--toast                     Notificación de Windows (requiere win10toast)
--toast_title STR           Título del toast (por defecto "Roblox Chat Guard")
```

---

## 📝 Ejemplos útiles

### Más robusto (HUD translúcido / texto pequeño)
```bash
 python roblox_chat_guard_local_rapidocr.py --video partida_roblox.mp4 --preview --preprocess light --scale 1.2 --gamma 1.1 --passes 1 --history 2 --fuzzy 0.83
```

### Frases personalizadas
```bash
python nemi.py --video partida_roblox.mp4 --preview   --forbidden "te deseo,mandame fotos,ven a mi casa,pack"
```
o desde archivo:
```bash
python nemi.py --video partida_roblox.mp4 --preview   --forbidden_file prohibidas.txt
```

---

## 📤 Salidas
- **Video anotado**: `chat_guard_local_annotated.mp4` (ROI en rojo si hay alerta, cajas OCR en verde).
- **CSV**: `chat_alertas_local.csv` (`timestamp`, `frase`, `linea`).
- **Log**: `chat_alertas_local.txt` (una línea por alerta).
- **Snapshots**: carpeta `hits/` (captura del frame cuando se dispara una alerta).

---

## 🧪 Tuning rápido (cheatsheet)
- Borroso/comprimido:  
  `--preprocess strong --scale 2.0 --gamma 1.2 --passes 2 --history 3 --fuzzy 0.8 --min_conf 0.55`
- Nítido 1440p/4K:  
  `--preprocess light --scale 1.2 --gamma 1.1 --passes 1 --history 2 --fuzzy 0.83`
- Texto claro sobre fondo claro: añade `--invert`
- Pocas detecciones: baja `--min_conf` a 0.5–0.6 y `--fuzzy` a 0.78–0.8
- Sin ventana GUI: `--roi x,y,w,h --headless`

---

## 🛠️ Solución de problemas
- **No detecta nada**: revisa que el **ROI** cubra todo el chat; usa `--debug` y prueba `--preprocess strong` + `--scale 2.0`.
- **La ventana no abre**: usa modo headless con `--roi`.
- **Lento**: baja `--fps`, usa `--passes 1`, reduce el tamaño del ROI.
- **Notificaciones**: instala `win10toast` y añade `--toast`.

---

## 🔒 Nota
Este proyecto está pensado para **moderación y seguridad** en contenido de chat. Adecúa las frases a tu contexto y cumple con las normas de la plataforma.

---
