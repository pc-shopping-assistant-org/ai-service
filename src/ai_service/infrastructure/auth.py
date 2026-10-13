"""RS256 access tokens issued by identity-service; never trust forwarded owner headers."""

from time import monotonic
from uuid import UUID

import httpx
import jwt


class AuthenticationError(Exception):
    """No valid authenticated identity; do not expose token/provider details."""


class JwtAuthenticator:
    def __init__(
        self,
        client: httpx.AsyncClient,
        jwks_url: str,
        issuer: str,
        audience: str,
        cache_seconds: int = 300,
    ) -> None:
        self.client = client
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.audience = audience
        self.cache_seconds = cache_seconds
        self._keys: list[jwt.PyJWK] = []
        self._expires_at = 0.0

    async def authenticate(self, token: str) -> UUID:
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
                raise AuthenticationError()
            if monotonic() >= self._expires_at:
                response = await self.client.get(self.jwks_url)
                response.raise_for_status()
                self._keys = jwt.PyJWKSet.from_dict(response.json()).keys
                self._expires_at = monotonic() + self.cache_seconds
            matches = [
                key
                for key in self._keys
                if key.key_id == header["kid"]
                and key.key_type == "RSA"
                and key.public_key_use in (None, "sig")
                and key.algorithm_name == "RS256"
            ]
            if len(matches) != 1:
                raise AuthenticationError()
            claims = jwt.decode(
                token,
                matches[0].key,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=self.audience,
                options={
                    "require": [
                        "exp",
                        "iat",
                        "iss",
                        "aud",
                        "sub",
                        "accountId",
                        "type",
                        "status",
                    ]
                },
            )
            if claims["type"] != "access" or claims["status"] != "ACTIVE":
                raise AuthenticationError()
            return UUID(claims["accountId"])
        except (
            jwt.PyJWTError,
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            raise AuthenticationError() from exc
