"""One-command launcher: python lancer.py (Python 3.10–3.12)."""
from pathlib import Path
import os
import subprocess
import sys
import venv


def main():
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise SystemExit("Utiliser Python 3.10, 3.11 ou 3.12 pour cette démo.")
    folder = Path(__file__).resolve().parent
    environment = folder / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("[Démo] Création de l'environnement Python…", flush=True)
        venv.create(environment, with_pip=True)
    probe = subprocess.run([str(python), "-c",
        "from importlib.metadata import version; from packaging.version import Version; "
        "assert Version('0.1.4') <= Version(version('paradis-tts')) < Version('0.2')"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode:
        print("[Démo] Installation de Paradis-TTS (premier lancement)…", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(folder / "requirements.txt")], check=True)
    try:
        return subprocess.call([str(python), str(folder / "app.py"), *sys.argv[1:]])
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError:
        raise SystemExit("Installation échouée. Vérifier la connexion Internet, puis relancer la même commande.")
