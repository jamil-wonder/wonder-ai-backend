# Backend Main Chunks

`backend/main.py` now loads these files in numeric order. They were split from
the previous monolithic `main.py` at top-level Python boundaries, so route
decorators, globals, startup hooks, and helpers keep the same runtime order.

This is a safe first split. Future cleanup can move each chunk into named
FastAPI routers such as auth, user, blogs, content, public, admin, and phase5.
