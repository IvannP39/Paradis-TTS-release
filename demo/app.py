"""Local HTTP demo. Only dependency: the published paradis-tts package."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
import wave
import webbrowser

MAX_TEXT = 1000
MAX_BODY = 16384
FOLDER = Path(__file__).resolve().parent


class Voice:
    def __init__(self, model_folder=None):
        self.model_folder = model_folder
        self.model = None
        self.state = "loading"
        self.error = None
        self.lock = threading.Lock()
        self.generating = threading.Lock()

    def load(self):
        try:
            from paradis_tts import ParadisTTS
            model = ParadisTTS(model_folder=self.model_folder)
            with self.lock:
                self.model = model
                self.state = "ready"
        except Exception as error:
            print(f"[Démo] Chargement impossible : {error}", flush=True)
            with self.lock:
                self.error = str(error)
                self.state = "error"

    def status(self):
        with self.lock:
            return {"state": self.state, "error": self.error, "max_characters": MAX_TEXT}


def encode_wav(audio, sample_rate):
    # Conversion in memory; no generated files accumulate on disk.
    pcm = (audio.clip(-1, 1) * 32767).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, voice):
        self.voice = voice
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "ParadisDemo/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def respond(self, status, payload, content_type="application/json; charset=utf-8"):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if isinstance(payload, dict) else payload
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origins = {"http://" + host for host in hosts}
        if self.headers.get("Host") not in hosts or self.headers.get("Origin") not in origins | {None}:
            self.respond(403, {"error": "Cette démo accepte uniquement les requêtes locales."})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        if self.path == "/":
            self.respond(200, (FOLDER / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self.respond(200, self.server.voice.status())
        elif self.path == "/favicon.ico":
            self.respond(204, b"", "image/x-icon")
        else:
            self.respond(404, {"error": "Page introuvable."})

    def do_POST(self):
        if not self.local_request():
            return
        if self.path != "/api/synthesize":
            self.respond(404, {"error": "Page introuvable."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.respond(415, {"error": "Un corps JSON est attendu."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("Requête trop volumineuse ou vide.")
            payload = json.loads(self.rfile.read(length))
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Entrer une phrase à prononcer.")
            text = text.strip()
            if len(text) > MAX_TEXT:
                raise ValueError(f"Le texte est limité à {MAX_TEXT} caractères.")
        except (ValueError, UnicodeError, TimeoutError) as error:
            self.respond(400, {"error": str(error)})
            return
        voice = self.server.voice
        if voice.status()["state"] != "ready":
            self.respond(503, {"error": "Le modèle n'est pas encore prêt. Vérifier son état dans l'interface."})
            return
        if not voice.generating.acquire(blocking=False):
            self.respond(409, {"error": "Une génération est déjà en cours. Réessayer dans un instant."})
            return
        try:
            started = time.perf_counter()
            audio = voice.model(text)
            elapsed = time.perf_counter() - started
            duration = len(audio) / voice.model.sample_rate
            result = {"audio": encode_wav(audio, voice.model.sample_rate),
                      "generation_seconds": elapsed, "audio_seconds": duration,
                      "realtime_speed": duration / elapsed, "sample_rate": voice.model.sample_rate}
            print(f"[Démo] Voix générée en {elapsed:.3f} s pour {duration:.2f} s d'audio.", flush=True)
            self.respond(200, result)
        except ValueError as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            print(f"[Démo] Génération impossible : {error}", flush=True)
            self.respond(500, {"error": "La génération a échoué. Le détail est affiché dans le terminal."})
        finally:
            voice.generating.release()

    def log_message(self, message, *args):
        # Keep model/download and synthesis messages visible without polling noise.
        if args and str(args[0]).startswith("GET /api/status "):
            return
        super().log_message(message, *args)


def main():
    parser = argparse.ArgumentParser(description="Démo locale de Paradis-TTS")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--model-folder", type=Path, help="Dossier ONNX local, sinon téléchargement Hugging Face")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Le port doit être compris entre 1 et 65535.")
    voice = Voice(args.model_folder)
    try:
        server = Server(("127.0.0.1", args.port), voice)
    except OSError as error:
        raise SystemExit(f"Impossible d'ouvrir le port {args.port} : {error}. Essayer --port 7861.")
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"\n[Démo] Ouvrir {address}\n[Démo] Arrêt : Ctrl+C\n", flush=True)
    threading.Thread(target=voice.load, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Démo] Serveur arrêté.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
