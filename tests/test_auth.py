from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.auth import AuthError, Principal, decode_token, issue_token
from app.config import Settings

SETTINGS = Settings(jwt_secret="test-secret-that-is-long-enough-for-hs256")
ALICE = Principal(employee_id="Z1001", company_id="zentrix", role="employee")


def _claims(**overrides):
    now = datetime.now(timezone.utc)
    claims = {"sub": "Z1001", "company_id": "zentrix", "role": "employee",
              "iss": SETTINGS.jwt_issuer, "aud": SETTINGS.jwt_audience,
              "iat": now, "exp": now + timedelta(minutes=5)}
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


def test_round_trip():
    assert decode_token(issue_token(ALICE, SETTINGS), SETTINGS) == ALICE


@pytest.mark.parametrize("token_factory", [
    pytest.param(lambda: jwt.encode(_claims(), "some-other-secret-of-sufficient-length",
                                    algorithm="HS256"), id="wrong-secret"),
    pytest.param(lambda: jwt.encode(_claims(exp=datetime.now(timezone.utc) - timedelta(seconds=1)),
                                    SETTINGS.jwt_secret, algorithm="HS256"), id="expired"),
    pytest.param(lambda: jwt.encode(_claims(aud="someone-else"), SETTINGS.jwt_secret,
                                    algorithm="HS256"), id="wrong-audience"),
    pytest.param(lambda: jwt.encode(_claims(iss="evil-issuer"), SETTINGS.jwt_secret,
                                    algorithm="HS256"), id="wrong-issuer"),
    pytest.param(lambda: jwt.encode(_claims(company_id=None), SETTINGS.jwt_secret,
                                    algorithm="HS256"), id="no-company"),
    pytest.param(lambda: jwt.encode(_claims(company_id="Zentrix; DROP"), SETTINGS.jwt_secret,
                                    algorithm="HS256"), id="malformed-company"),
    pytest.param(lambda: jwt.encode(_claims(role="superuser"), SETTINGS.jwt_secret,
                                    algorithm="HS256"), id="unknown-role"),
    pytest.param(lambda: jwt.encode(_claims(), None, algorithm="none"), id="alg-none"),
    pytest.param(lambda: "not.a.jwt", id="garbage"),
])
def test_bad_tokens_are_rejected(token_factory):
    with pytest.raises(AuthError):
        decode_token(token_factory(), SETTINGS)


def test_tampered_payload_is_rejected():
    header, _, signature = issue_token(ALICE, SETTINGS).split(".")
    forged_payload = jwt.utils.base64url_encode(
        b'{"sub":"Z1001","company_id":"meridian","role":"employee"}').decode()
    with pytest.raises(AuthError):
        decode_token(f"{header}.{forged_payload}.{signature}", SETTINGS)
