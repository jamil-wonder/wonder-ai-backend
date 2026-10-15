"""H3: forgot / reset password.

  - A reset link carries a random token that works for ONE HOUR and ONCE.
    Only a hash of the token is stored, so a copy of the database can't be used
    to reset anyone's password.
  - Asking for a link answers the same thing whether or not the email has an
    account, so the form can't be used to find out who is registered.
  - At most 5 links per account per hour.
"""
import hashlib
import secrets
from datetime import datetime

RESET_TTL_MINUTES = 60
MAX_REQUESTS_PER_HOUR = 5
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72          # bcrypt silently ignores (or rejects) anything longer

GENERIC_MESSAGE = (
    "If an account exists for that email, we've sent a link to reset the password. "
    "It works for 1 hour."
)
INVALID_LINK_MESSAGE = "This reset link has expired or was already used. Please ask for a new one."
DONE_MESSAGE = "Your password has been changed. You can sign in with it now."


def new_token() -> tuple[str, str]:
    """(token to put in the email link, hash to store)."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


def hash_token(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def token_state(doc: dict | None, now: datetime) -> str:
    """"ok" | "missing" | "used" | "expired"."""
    if not doc:
        return "missing"
    if doc.get("used_at") is not None:
        return "used"
    expires = doc.get("expires_at")
    if not isinstance(expires, datetime) or expires <= now:
        return "expired"
    return "ok"


def password_problem(password) -> str:
    """"" if the new password is acceptable, else what to tell the user."""
    pw = str(password or "")
    if len(pw) < MIN_PASSWORD_LENGTH:
        return f"Choose a password with at least {MIN_PASSWORD_LENGTH} characters."
    if len(pw.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return "That password is too long. Please use 72 characters or fewer."
    if not pw.strip():
        return "Choose a password that isn't only spaces."
    return ""


def build_reset_email(name: str, reset_url: str) -> tuple[str, str]:
    """(html, text) for the reset email."""
    display = (name or "there").strip() or "there"
    text = (
        f"Hi {display},\n\n"
        f"Someone asked to reset the password for your Wonderscore account. "
        f"Open this link to choose a new one (it works for {RESET_TTL_MINUTES // 60} hour, once):\n\n"
        f"{reset_url}\n\n"
        f"If you didn't ask for this, you can ignore this email - your password stays as it is.\n"
    )
    html = f"""
    <div style="margin:0;padding:0;background:#faf8f3;">
      <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;padding:28px 16px;color:#23211b;">
        <div style="background:#ffffff;border:1px solid #ece3d1;border-radius:24px;padding:30px 28px;">
          <div style="font-size:20px;font-weight:700;color:#15463b;margin-bottom:18px;">Wonderscore</div>
          <h1 style="margin:0 0 10px 0;font-size:24px;line-height:30px;color:#15463b;">Reset your password</h1>
          <p style="margin:0 0 18px 0;font-size:14px;line-height:22px;color:#4a4437;">Hi {display}, someone asked to reset the password for your account. This link works for {RESET_TTL_MINUTES // 60} hour, and only once.</p>
          <p style="margin:0 0 22px 0;"><a href="{reset_url}" style="display:inline-block;background:#15463b;color:#ffffff;text-decoration:none;font-weight:700;font-size:14px;padding:12px 22px;border-radius:12px;">Choose a new password</a></p>
          <p style="margin:0 0 6px 0;font-size:12px;line-height:18px;color:#8a8273;">If the button doesn't work, copy this address into your browser:</p>
          <p style="margin:0 0 18px 0;font-size:12px;line-height:18px;color:#6f6757;word-break:break-all;">{reset_url}</p>
          <p style="margin:0;font-size:12px;line-height:18px;color:#8a8273;">If you didn't ask for this, ignore this email - your password stays as it is.</p>
        </div>
      </div>
    </div>
    """
    return html, text
