from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from ai_service.infrastructure.auth import AuthenticationError, JwtAuthenticator


@pytest.mark.asyncio
async def test_rs256_identity_rejects_refresh_expired_and_wrong_audience():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    jwk.update(kid="test", use="sig", alg="RS256")
    account_id = uuid4()
    claims = {
        "sub": "customer@example.test",
        "accountId": str(account_id),
        "status": "ACTIVE",
        "type": "access",
        "iss": "http://identity-service",
        "aud": "pc-shopping-api",
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(minutes=1),
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"keys": [jwk]})
        )
    ) as client:
        auth = JwtAuthenticator(
            client,
            "http://identity.test/jwks",
            "http://identity-service",
            "pc-shopping-api",
        )
        token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test"})
        assert await auth.authenticate(token) == account_id
        for change in [
            {"type": "REFRESH"},
            {"aud": "wrong"},
            {"status": "LOCKED"},
            {"exp": datetime.now(UTC) - timedelta(seconds=10)},
        ]:
            bad = jwt.encode(
                claims | change, key, algorithm="RS256", headers={"kid": "test"}
            )
            with pytest.raises(AuthenticationError):
                await auth.authenticate(bad)
        with pytest.raises(AuthenticationError):
            await auth.authenticate(jwt.encode(claims, "s" * 32, algorithm="HS256"))
