"""Is Gemini working for the key this machine uses? Prints the real answer from
Google (never the key). Run it from the backend folder:

    .venv\\Scripts\\python.exe check_gemini.py        (Windows)
    .venv/bin/python check_gemini.py                  (Linux / the server)

Share the whole output when reporting a Gemini problem."""
import hashlib
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from utils.gemini_utils import describe_gemini_error, get_model_chain  # noqa: E402


def main() -> int:
    key = os.getenv("GEMINI_API_KEY", "")
    models = get_model_chain()
    print(f"key: {'(not set)' if not key else 'set, fingerprint ' + hashlib.sha256(key.encode()).hexdigest()[:10] + ', length ' + str(len(key))}")
    print(f"model: {models[0]}")
    if not key:
        print("RESULT: no GEMINI_API_KEY in this environment")
        return 2
    client = genai.Client(api_key=key)
    failed = 0
    for label, config in (
        ("plain call   ", types.GenerateContentConfig(max_output_tokens=20, temperature=0.0)),
        ("grounded call", types.GenerateContentConfig(max_output_tokens=60, temperature=0.0, tools=[{"google_search": {}}])),
    ):
        started = time.time()
        try:
            client.models.generate_content(model=models[0], contents="Name one e-commerce platform used in Bangladesh.", config=config)
            print(f"{label}: OK in {time.time() - started:.1f}s")
        except Exception as exc:  # noqa: BLE001 - we want whatever Google said
            failed += 1
            print(f"{label}: FAILED in {time.time() - started:.1f}s")
            print(f"   {describe_gemini_error(exc)}")
    print("RESULT:", "Gemini works with this key" if not failed else f"{failed} of 2 calls failed - see the lines above")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
