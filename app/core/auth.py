"""Authentication and RBAC.

AUTH_MODE=firebase  verify Firebase ID tokens; role from the `role` claim, else ADMIN_EMAILS, else DEFAULT_ROLE
AUTH_MODE=jwt       HS256 tokens signed with JWT_SECRET (dev/CI); Google sign-in also works if Firebase is configured
AUTH_MODE=off       no auth, everyone is admin (refused when APP_ENV=prod)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings

ROLES = {"agent": 1, "admin": 2}


@dataclass
class User:
    uid: str
    role: str = "agent"
    email: str | None = None
    name: str | None = None
    provider: str = "local"      # local (dev token) | google.com | password | ...


_firebase_ready = False


def _firebase_claims(token: str, s: Settings) -> dict:
    """Verify a Firebase ID token (Google sign-in, email/password, ...) and return its claims."""
    global _firebase_ready
    import firebase_admin  # lazy: only needed when Firebase is configured
    from firebase_admin import auth as fb_auth
    from firebase_admin import credentials

    if not _firebase_ready:
        if not firebase_admin._apps:
            cred = credentials.Certificate(s.firebase_credentials_file) if s.firebase_credentials_file else None
            project = s.effective_firebase_project
            firebase_admin.initialize_app(cred, {"projectId": project} if project else None)
        _firebase_ready = True
    return fb_auth.verify_id_token(token, check_revoked=False)


def role_for(claims: dict, s: Settings) -> str:
    """RBAC for federated users. A `role` custom claim (set with the Admin SDK) always wins; otherwise a *verified*
    email listed in ADMIN_EMAILS is an admin; everyone else gets DEFAULT_ROLE (agent)."""
    claimed = claims.get("role")
    if claimed in ROLES:
        return claimed
    email = (claims.get("email") or "").lower()
    if email and claims.get("email_verified") and email in s.admin_email_list:
        return "admin"
    return s.default_role


def user_from_claims(claims: dict, s: Settings) -> User:
    email = (claims.get("email") or "").lower() or None
    domains = s.allowed_domain_list
    if domains and (not email or email.rsplit("@", 1)[-1] not in domains):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"sign-in is limited to: {', '.join(domains)}")
    uid = claims.get("uid") or claims.get("user_id") or claims.get("sub")
    provider = (claims.get("firebase") or {}).get("sign_in_provider", "firebase")
    return User(uid=uid, role=role_for(claims, s), email=email, name=claims.get("name"), provider=provider)


def _verify_firebase(token: str, s: Settings) -> User:
    return user_from_claims(_firebase_claims(token, s), s)


def mint_dev_token(uid: str, role: str, s: Settings, ttl_s: int = 3600) -> str:
    now = int(time.time())
    return jwt.encode({"sub": uid, "role": role, "iat": now, "exp": now + ttl_s}, s.jwt_secret, algorithm="HS256")


def get_current_user(authorization: str | None = Header(default=None), s: Settings = Depends(get_settings)) -> User:
    if s.auth_mode == "off":
        if s.app_env == "prod":
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "AUTH_MODE=off is not allowed in prod")
        return User(uid="dev-user", role="admin")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    token = authorization.split(" ", 1)[1].strip()
    try:
        if s.auth_mode == "jwt":
            try:
                c = jwt.decode(token, s.jwt_secret, algorithms=["HS256"])
                return User(uid=c["sub"], role=c.get("role", "agent"), email=c.get("email"))
            except jwt.PyJWTError:
                if not s.firebase_enabled:
                    raise
                return _verify_firebase(token, s)   # not a dev token: maybe a Google/Firebase ID token
        return _verify_firebase(token, s)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"}) from None


def require_role(min_role: str):
    def dep(user: User = Depends(get_current_user)) -> User:
        if ROLES.get(user.role, 0) < ROLES[min_role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires role {min_role}")
        return user
    return dep
