"""Background-job worker process entrypoint.

Runs the same job logic as the web API (imported via ``main``, which execs
all of ``main_parts/*``) but as a separate process with no HTTP server, so
redeploying the web container never interrupts an in-flight Sunday/Wednesday
/Phase-5 job. See docs/infra-diagnosis.html for the incident this fixes.
"""
import asyncio

import main

if __name__ == "__main__":
    asyncio.run(main.run_worker())
