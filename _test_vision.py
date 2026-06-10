"""Quick test: does moondream handle images via Ollama API?"""
import base64, json, urllib.request, sys
sys.path.insert(0, '.')
from smart_glasses import require_cv2

cv2 = require_cv2()
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()

if not ok:
    print("Failed to capture frame")
    sys.exit(1)

_, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
jpeg = encoded.tobytes()
print(f"Frame: {frame.shape}, JPEG: {len(jpeg)} bytes")

for model in ["moondream:latest", "llava:7b"]:
    payload = {
        "model": model,
        "prompt": "Describe this image in one short sentence.",
        "stream": False,
        "images": [base64.b64encode(jpeg).decode("ascii")],
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "").strip()
        print(f"\n{model}: [{len(text)} chars] {text[:200]}")
    except Exception as e:
        print(f"\n{model}: ERROR {e}")
