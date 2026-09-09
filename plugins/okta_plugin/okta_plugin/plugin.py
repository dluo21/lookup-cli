"""
Okta connector: account status for one person.

Endpoint: `GET {OKTA_ORG_URL}/api/v1/users/{login}` with an
`Authorization: SSWS <token>` header.

**Not-found semantics.** A user Okta has never heard of returns a
*successful* result with `data["found"] is False`, not `error=`. The guide
asks each connector to decide this explicitly: for an offboarding lookup,
"this person has no Okta account" is a real answer, whereas an `error=`
would make `UnifiedRecord.field_for("okta")` return None and be
indistinguishable from "Okta was unreachable".

**Devices.** `fetch_devices()` (surfaced as `okta <user> -d`) lists the
devices Okta associates with a user, via
`GET /api/v1/users/{userId}/devices`. Scope caveat worth repeating to
users: this is Okta's own device registry -- machines enrolled through
Okta Verify / device trust -- and NOT the Jamf hardware inventory that
Stage 4 will add. Someone can hold a laptop that Okta has never
seen. It is deliberately a separate call, not part of `fetch()`, because
Stage 7 runs `fetch()` for every plugin on every lookup and shouldn't pay
for a second round trip nobody asked for.

**Applications.** `fetch_apps()` (`okta <user> -a`) reads
`GET /api/v1/users/{userId}/appLinks`, the list behind the user's Okta
dashboard. It answers "what can this person open", not "how were they
granted it" -- direct-vs-group assignment lives on
`/apps/{appId}/users/{userId}` and would cost one request per app, so it
is not fetched. Hidden tiles are included: a hidden app is still an
assignment, and an offboarding check that skipped them would under-report.

**Authenticators.** `fetch_authenticators()` (`okta <user> -u`) reads
`GET /api/v1/users/{userId}/factors`. The API says "factor", the Okta
admin console says "authenticator"; the CLI follows the console and the
API's word is kept for anything touching the wire. `profile.questionText`
is deliberately dropped -- that a security question is enrolled is the
useful fact, while the question itself is a recovery-credential hint with
no operational value here. Phone numbers and emails are shown exactly as
Okta returns them, which is already partially masked for SMS.

**Name search.** `fetch_search()` (`okta --find <name>`) resolves a partial
name to a username via `GET /api/v1/users?search=...`, for when someone
knows a colleague's first or surname but not their login. `--find` is
long-only and composes with nothing: short flags here are section
selectors, and search is not a section -- it answers "who is this person",
not "what do you want to see about them". Results are never cached, and no
status filter is sent (see `build_search_expression`).

Required env vars (see `.env.example`):
    OKTA_ORG_URL        e.g. https://acme.okta.com
    OKTA_API_TOKEN      an SSWS token
Optional:
    OKTA_TIMEOUT_SECONDS        per-request timeout (default 10)
    OKTA_ACCESS_ATTRIBUTE       custom profile attribute carrying this org's
                                access decision (default `access_blocked`)
    LOOKUP_CLI_MOCK_OKTA=1      serve a fixture instead of calling out

**Custom profile attribute.** This org's Universal Directory defines an
attribute displayed in the Profile Editor as "ACCESS BLOCKED", variable
name `access_blocked`. It arrives inside the `profile` object of the user
payload we already fetch, so reading it costs no extra request. Its value
is reported verbatim -- a boolean stays `true`/`false` rather than becoming
yes/no -- so an operator sees exactly what the Okta admin UI shows. Only
the one configured attribute is read: Okta profiles routinely carry
manager, employee id and personal contact details, and everything in
`data` is written to the plaintext local cache.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx
import typer
from rich.console import Console
from rich.table import Table

from lookup_cli.plugins.base import ConnectorPlugin, ConnectorResult
from lookup_cli.redaction import safe_error

DEFAULT_TIMEOUT_SECONDS = 10.0

#: Okta statuses that mean the account can actually be used.
_ACTIVE_STATUSES = frozenset({"ACTIVE"})

#: Hard bound on Link-header following, so a looping or malformed `next`
#: can't hang the CLI. Far above any real user's device count.
_MAX_PAGES = 20

#: Custom Universal Directory attribute carrying this org's access decision.
#: Shown in the Okta Profile Editor as "ACCESS BLOCKED" with the variable
#: name `access_blocked`. Custom attribute names are org-specific, so the
#: name is overridable via OKTA_ACCESS_ATTRIBUTE rather than hardcoded.
DEFAULT_ACCESS_ATTRIBUTE = "access_blocked"

#: Label for that value in CLI output.
ACCESS_FIELD_LABEL = "access blocked"

#: Okta retains System Log data for roughly 90 days. Asking for longer cannot
#: return longer, and letting a caller believe otherwise would put a window in
#: the column header that the data does not actually cover.
MAX_LOG_WINDOW = timedelta(days=90)

#: Sign-in events that can carry device identity.
_SIGNIN_EVENT_TYPES = ("user.session.start", "user.authentication.sso")

_SINCE_RE = re.compile(r"^\s*(\d+)\s*([dh])\s*$", re.IGNORECASE)

#: Profile fields a searcher might plausibly know. `sw` (startsWith) is what
#: Okta's `search` supports broadly; it means "dennis" finds Dennis but "enn"
#: finds nobody, which is an acceptable trade for covering "I know their first
#: or surname".
_SEARCH_FIELDS = ("profile.firstName", "profile.lastName", "profile.login", "profile.email")

#: One page, requested once. Search is interactive, not an audit -- following
#: Link headers to page thousands of users to render a 15-row table would burn
#: rate-limit budget for nothing. A full page back is reported as truncated
#: rather than passed off as the complete answer.
MAX_SEARCH_RESULTS = 200

#: Rows the chooser prints before it starts saying "N more".
MAX_MATCHES_SHOWN = 15


def build_search_expression(query: str) -> str:
    """Okta `search` expression matching `query` against the name fields.

    **Whitespace splits into AND-ed groups.** Each token must match *some*
    field, so "dennis luo" is `(any field starts with dennis) and (any field
    starts with luo)` -- which also means token order doesn't matter, and
    nobody has to know whether the directory stores "Dennis Luo" or "Luo,
    Dennis". Treating the whole string as one term instead made a full-name
    search match nobody, which was doubly bad because adding a surname is
    exactly the advice the "too many matches" message gives.

    **No status clause, deliberately.** Okta's List Users endpoint excludes
    `DEPROVISIONED` users by default, and any status predicate added here
    risks reproducing that exclusion. For a tool whose central question is
    "did this person's access actually get revoked", a search that silently
    omits the deactivated person is worse than no search -- it looks
    complete. Whether `search=` itself inherits the default exclusion is
    server-side behaviour that mocks cannot prove; it is flagged in
    docs/STAGES.md for the live smoke test, and if it does, the fix is an
    explicit all-statuses clause added here.
    """
    tokens = (query or "").split()
    if not tokens:
        raise ValueError("--find needs a name to search for.")

    groups = []
    for token in tokens:
        # Escape backslashes then quotes, so a name can't terminate the
        # filter string or smuggle an operator into it.
        safe = token.replace("\\", "\\\\").replace('"', '\\"')
        groups.append(" or ".join(f'{field} sw "{safe}"' for field in _SEARCH_FIELDS))

    if len(groups) == 1:
        return groups[0]
    # Parenthesised: `a or b and c or d` binds wrongly and the AND would
    # silently stop narrowing.
    return " and ".join(f"({group})" for group in groups)


def parse_since(raw: str) -> timedelta:
    """Parse a `--since` window like `90d` or `12h`, clamped to retention."""
    match = _SINCE_RE.match(raw or "")
    if not match:
        raise ValueError(
            f"could not parse --since {raw!r}. Use a number followed by "
            f"'d' (days) or 'h' (hours), e.g. 30d or 12h."
        )
    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError("--since must be greater than zero.")
    window = timedelta(days=amount) if match.group(2).lower() == "d" else timedelta(hours=amount)
    return min(window, MAX_LOG_WINDOW)


def describe_window(window: timedelta) -> str:
    """Short label for a window, for the column header."""
    if window >= timedelta(days=1) and window.total_seconds() % 86400 == 0:
        return f"{int(window.total_seconds() // 86400)}d"
    return f"{int(window.total_seconds() // 3600)}h"

#: Okta's status enum has eight values, and the raw name is not always what
#: an operator needs to read. "Deactivated" in the Okta admin UI means
#: DEPROVISIONED specifically -- SUSPENDED also blocks login but is a
#: different state, and conflating them would mislead someone checking
#: whether an offboarding actually completed.
_DEACTIVATED_STATUS = "DEPROVISIONED"


def format_profile_value(raw: object) -> str:
    """Render a profile attribute exactly as Okta returned it.

    No interpretation: a boolean stays a boolean rather than becoming
    yes/no, so an operator sees the same value the Okta admin UI shows.
    `json.dumps` rather than `str` for non-strings, because Okta's JSON says
    `true` while Python's `str(True)` says `True`.

    An absent or null attribute has nothing to render verbatim, so it falls
    back to the table's usual empty marker -- which keeps it distinct from
    an explicit `false`.
    """
    if raw is None:
        return "-"
    if isinstance(raw, str):
        return raw
    return json.dumps(raw)


#: Okta's `factorType` values are wire identifiers, not something an operator
#: should have to decode -- `token:software:totp` is the clearest example.
#: Anything absent falls through to the raw value: Okta keeps adding
#: authenticator types, and inventing a label for one we don't recognise
#: would be worse than showing what the API actually said.
_FACTOR_LABELS: dict[str, str] = {
    "push": "Okta Verify push",
    "signed_nonce": "Okta FastPass",
    "webauthn": "WebAuthn / passkey",
    "u2f": "Security key (U2F)",
    "sms": "SMS",
    "call": "Voice call",
    "email": "Email",
    "question": "Security question",
    "token:software:totp": "TOTP app",
    "token:hardware": "Hardware token",
    "token": "Token",
    "password": "Password",
}

#: Profile keys that identify *which* authenticator this is, best first. A
#: named device beats the credential id behind it, because the name is what an
#: operator recognises. `questionText` is deliberately absent -- see the module
#: docstring.
_FACTOR_DETAIL_FIELDS = ("name", "authenticatorName", "phoneNumber", "email", "credentialId")

#: Anything unrecognised stays yellow rather than green: an unknown state is
#: not evidence that an authenticator is fine.
_FACTOR_STATUS_COLOURS: dict[str, str] = {
    "ACTIVE": "green",
    "PENDING_ACTIVATION": "yellow",
    "NOT_SETUP": "yellow",
    "INACTIVE": "red",
    "DISABLED": "red",
    "EXPIRED": "red",
}


def factor_label(factor_type: str | None, provider: str | None) -> str | None:
    """Readable name for an authenticator, keeping a non-Okta provider visible.

    A Duo push and an Okta Verify push are different systems to go and revoke,
    so the provider is named whenever it isn't Okta's own.
    """
    if not factor_type:
        return None
    label = _FACTOR_LABELS.get(factor_type, factor_type)
    if provider and provider.upper() != "OKTA":
        label = f"{label} ({provider})"
    return label


def factor_detail(profile: dict | None) -> str | None:
    """The most identifying value on a factor profile, or None."""
    for key in _FACTOR_DETAIL_FIELDS:
        value = (profile or {}).get(key)
        if value:
            return str(value)
    return None


_STATUS_NOTES: dict[str, tuple[str, str]] = {
    "ACTIVE": ("green", ""),
    "DEPROVISIONED": ("red", "deactivated"),
    "SUSPENDED": ("red", "suspended"),
    "LOCKED_OUT": ("yellow", "locked out"),
    "PASSWORD_EXPIRED": ("yellow", "password expired"),
    "RECOVERY": ("yellow", "in password recovery"),
    "STAGED": ("yellow", "not yet activated"),
    "PROVISIONED": ("yellow", "activation pending"),
}


class OktaPlugin(ConnectorPlugin):
    name = "okta"
    required_credentials = ("OKTA_ORG_URL", "OKTA_API_TOKEN")

    @property
    def _access_attribute(self) -> str:
        return self.config.get("OKTA_ACCESS_ATTRIBUTE") or DEFAULT_ACCESS_ATTRIBUTE

    async def fetch(self, identifier: str) -> ConnectorResult:
        try:
            raw = await self._call_backend(identifier)
        except Exception as exc:  # noqa: BLE001 - contract: never crash aggregation
            return ConnectorResult(
                plugin_name=self.name,
                identifier=identifier,
                # The token is passed explicitly as well as being picked up
                # from the environment: in mock/test runs it may only exist
                # in the injected config.
                error=safe_error(exc, secrets=[self.config.get("OKTA_API_TOKEN")]),
            )

        if raw is None:
            return ConnectorResult(
                plugin_name=self.name,
                identifier=identifier,
                data={"found": False, "status": None},
                tags=["not-found"],
            )

        return self._to_result(identifier, raw)

    async def fetch_devices(self, identifier: str, *, okta_id: str | None = None) -> ConnectorResult:
        """List the devices Okta associates with `identifier`.

        Pass `okta_id` when the caller already resolved the user (as the CLI
        does) to skip a redundant lookup. Like `fetch()`, this never raises
        for ordinary failures.
        """
        try:
            okta_id = await self._resolve_okta_id(identifier, okta_id)
            if okta_id is None:
                return ConnectorResult(
                    plugin_name=self.name,
                    identifier=identifier,
                    data={"found": False, "devices": [], "count": 0},
                    tags=["not-found"],
                )

            raw_devices = await self._call_devices_backend(okta_id)
        except Exception as exc:  # noqa: BLE001 - contract: never crash aggregation
            return ConnectorResult(
                plugin_name=self.name,
                identifier=identifier,
                error=safe_error(exc, secrets=[self.config.get("OKTA_API_TOKEN")]),
            )

        devices = [self._to_device(entry) for entry in raw_devices]
        return ConnectorResult(
            plugin_name=self.name,
            identifier=identifier,
            data={"found": True, "devices": devices, "count": len(devices)},
            properties={"okta_id": okta_id},
            tags=["no-devices"] if not devices else ["has-devices"],
        )

    async def fetch_apps(self, identifier: str, *, okta_id: str | None = None) -> ConnectorResult:
        """List the applications assigned to `identifier` in Okta.

        Answers "what can this person open", not "how were they granted it" --
        see the module docstring. Like `fetch()`, never raises for ordinary
        failures.
        """
        try:
            resolved = await self._resolve_okta_id(identifier, okta_id)
            if resolved is None:
                return ConnectorResult(
                    plugin_name=self.name,
                    identifier=identifier,
                    data={"found": False, "apps": [], "count": 0},
                    tags=["not-found"],
                )
            raw_apps = await self._call_apps_backend(resolved)
        except Exception as exc:  # noqa: BLE001 - contract: never crash aggregation
            return ConnectorResult(
                plugin_name=self.name,
                identifier=identifier,
                error=safe_error(exc, secrets=[self.config.get("OKTA_API_TOKEN")]),
            )

        # Okta returns dashboard sort order, which is per-user and arbitrary.
        # Alphabetical means two people's app lists can actually be compared.
        apps = sorted(
            (self._to_app(entry) for entry in raw_apps),
            key=lambda app: (app["label"] or "").lower(),
        )
        return ConnectorResult(
            plugin_name=self.name,
            identifier=identifier,
            data={"found": True, "apps": apps, "count": len(apps)},
            properties={"okta_id": resolved},
            tags=["no-apps"] if not apps else ["has-apps"],
        )

    async def fetch_authenticators(
        self, identifier: str, *, okta_id: str | None = None
    ) -> ConnectorResult:
        """List the authenticators (API: "factors") enrolled by `identifier`.

        Includes inactive and half-finished enrolments: a disabled
        authenticator is still enrolled, and an offboarding check wants the
        whole picture rather than only what currently works.
        """
        try:
            resolved = await self._resolve_okta_id(identifier, okta_id)
            if resolved is None:
                return ConnectorResult(
                    plugin_name=self.name,
                    identifier=identifier,
                    data={"found": False, "authenticators": [], "count": 0},
                    tags=["not-found"],
                )
            raw_factors = await self._call_factors_backend(resolved)
        except Exception as exc:  # noqa: BLE001 - contract: never crash aggregation
            return ConnectorResult(
                plugin_name=self.name,
                identifier=identifier,
                error=safe_error(exc, secrets=[self.config.get("OKTA_API_TOKEN")]),
            )

        factors = sorted(
            (self._to_factor(entry) for entry in raw_factors),
            key=lambda f: ((f["label"] or "").lower(), (f["detail"] or "").lower()),
        )
        return ConnectorResult(
            plugin_name=self.name,
            identifier=identifier,
            data={"found": True, "authenticators": factors, "count": len(factors)},
            properties={"okta_id": resolved},
            tags=["no-authenticators"] if not factors else ["has-authenticators"],
        )

    async def fetch_search(self, query: str, *, fetch_all: bool = False) -> ConnectorResult:
        """Find users whose name, login or email starts with `query`.

        Returns candidates and never picks one: a single hit is not the same
        claim as the right person, and the section flags act on whoever the
        operator names next. Like `fetch()`, never raises for ordinary
        failures.
        """
        try:
            expression = build_search_expression(query)
            raw_users, truncated = await self._call_search_backend(expression, fetch_all)
        except ValueError as exc:
            # Bad input, not a service failure -- but still an error result
            # rather than an exception, per the connector contract.
            return ConnectorResult(plugin_name=self.name, identifier=query, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - contract: never crash aggregation
            return ConnectorResult(
                plugin_name=self.name,
                identifier=query,
                error=safe_error(exc, secrets=[self.config.get("OKTA_API_TOKEN")]),
            )

        matches = sorted(
            (self._to_match(user) for user in raw_users),
            key=lambda m: (m["login"] or "").lower(),
        )
        return ConnectorResult(
            plugin_name=self.name,
            identifier=query,
            data={
                "matches": matches,
                "count": len(matches),
                # Without --all: a full page back means Okta may be holding
                # more. With --all: we ran out of page budget while a `next`
                # link still existed. Either way the answer is incomplete, and
                # saying nothing would let it read as "that is everyone".
                "truncated": truncated,
                # Read by Stage 7's cache integration. A cached hit could
                # report someone ACTIVE minutes after they were deactivated --
                # wrong in exactly the case that matters.
                "cacheable": False,
            },
            tags=["no-matches"] if not matches else ["has-matches"],
        )

    async def fetch_device_signins(
        self,
        okta_id: str,
        *,
        since: timedelta,
        device_ids: set[str] | None = None,
    ) -> ConnectorResult:
        """Most recent successful sign-in per device, from the System Log.

        `/users/{id}/devices` has no last-login field -- its `lastUpdated`
        tracks changes to the device *record*, not sign-ins -- so this is a
        separate source correlated on `device.id`.

        Pass `device_ids` when the caller knows which devices it cares about:
        results come back newest-first, so once every device has been seen the
        remaining pages cannot change the answer and paging stops early. That
        matters because /api/v1/logs is Okta's most rate-limited endpoint.
        """
        try:
            signins = await self._call_logs_backend(okta_id, since, device_ids)
        except Exception as exc:  # noqa: BLE001 - contract: never crash aggregation
            return ConnectorResult(
                plugin_name=self.name,
                identifier=okta_id,
                error=safe_error(exc, secrets=[self.config.get("OKTA_API_TOKEN")]),
            )

        return ConnectorResult(
            plugin_name=self.name,
            identifier=okta_id,
            data={"signins": signins, "window": describe_window(since)},
        )

    # -- backend seam ---------------------------------------------------------

    async def _call_logs_backend(
        self, okta_id: str, since: timedelta, device_ids: set[str] | None
    ) -> dict[str, str]:
        if self.mock_mode:
            return self._mock_signins_fixture()

        org_url = self.config.require("OKTA_ORG_URL").rstrip("/")
        event_filter = " or ".join(f'eventType eq "{e}"' for e in _SIGNIN_EVENT_TYPES)
        params = {
            "since": (datetime.now(timezone.utc) - since).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "filter": f'actor.id eq "{okta_id}" and ({event_filter})',
            "sortOrder": "DESCENDING",
            "limit": "1000",
        }

        url = f"{org_url}/api/v1/logs"
        signins: dict[str, str] = {}
        seen_urls: set[str] = set()
        first = True

        async with self._client() as client:
            for _ in range(_MAX_PAGES):
                if url in seen_urls:
                    break
                seen_urls.add(url)

                response = await client.get(
                    url, headers=self._headers(), params=params if first else None
                )
                first = False
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        "Okta refused the System Log request. This API token may lack "
                        "System Log read access, which is granted separately from user "
                        "read access."
                    )
                response.raise_for_status()

                for event in response.json():
                    if (event.get("outcome") or {}).get("result") != "SUCCESS":
                        continue
                    device_id = (event.get("device") or {}).get("id")
                    if not device_id:
                        # Not every auth event stamps a device. Guessing from
                        # the user agent could not tell two MacBooks apart, so
                        # an unattributable event is dropped rather than
                        # assigned to the wrong machine.
                        continue
                    # DESCENDING: the first sighting is the most recent.
                    signins.setdefault(device_id, event.get("published"))

                if device_ids and device_ids.issubset(signins):
                    break

                next_url = response.links.get("next", {}).get("url")
                if not next_url:
                    break
                url = next_url

        return signins

    def _mock_signins_fixture(self) -> dict[str, str]:
        return {"guoMOCK00000000000001": "2026-01-01T09:15:00.000Z"}


    async def _resolve_okta_id(self, identifier: str, okta_id: str | None) -> str | None:
        """Turn a login into an Okta id, or None when there's no such user.

        Callers that already resolved the user (as the CLI does when several
        sections are on screen) pass `okta_id` and skip the round trip.
        """
        if okta_id is not None:
            return okta_id
        user = await self._call_backend(identifier)
        return None if user is None else user.get("id")

    def _user_url(self, okta_id: str, suffix: str) -> str:
        org_url = self.config.require("OKTA_ORG_URL").rstrip("/")
        # quote(safe="") so an id or login can't walk off /api/v1/users.
        return f"{org_url}/api/v1/users/{quote(okta_id, safe='')}/{suffix}"

    async def _fetch_all_pages(
        self, url: str, *, on_404: str | None = None, on_denied: str | None = None
    ) -> list[dict]:
        """Follow Okta's `Link: rel="next"` pagination and concatenate pages.

        Shared by every list endpoint here. Under-reporting someone's devices,
        apps or authenticators is the worst failure this tool has, so stopping
        at page one is never the answer -- and one implementation means the
        three call sites can't quietly drift apart.

        `on_404` / `on_denied` turn a status code into an actionable message
        for endpoints where the generic HTTP error would mislead.
        """
        entries: list[dict] = []
        seen_urls: set[str] = set()

        async with self._client() as client:
            for _ in range(_MAX_PAGES):
                if url in seen_urls:
                    break  # self-referential `next`; stop rather than loop
                seen_urls.add(url)

                response = await client.get(url, headers=self._headers())
                if response.status_code == 404 and on_404:
                    raise RuntimeError(on_404)
                if response.status_code in (401, 403) and on_denied:
                    raise RuntimeError(on_denied)
                response.raise_for_status()

                entries.extend(response.json())

                next_url = response.links.get("next", {}).get("url")
                if not next_url:
                    break
                url = next_url

        return entries

    async def _call_devices_backend(self, okta_id: str) -> list[dict]:
        if self.mock_mode:
            return self._mock_devices_fixture()

        return await self._fetch_all_pages(
            self._user_url(okta_id, "devices"),
            # The user exists (we just resolved them), so a 404 here means the
            # device API itself is unavailable -- typically an Okta Classic
            # org. Say that, don't say "not found".
            on_404=(
                "Okta returned 404 for the device endpoint. This org may not "
                "have Okta Identity Engine device management enabled, or the "
                "API token may lack the devices scope."
            ),
        )

    async def _call_search_backend(
        self, expression: str, fetch_all: bool = False
    ) -> tuple[list[dict], bool]:
        """Return (users, truncated).

        One page by default: search is interactive, not an audit, and paging
        thousands of users to render a 15-row table would burn rate-limit
        budget nobody asked to spend. `--all` opts into the extra requests.
        """
        if self.mock_mode:
            return self._mock_search_fixture(), False

        org_url = self.config.require("OKTA_ORG_URL").rstrip("/")
        url = f"{org_url}/api/v1/users"
        params: dict[str, str] | None = {
            "search": expression,
            "limit": str(MAX_SEARCH_RESULTS),
        }

        users: list[dict] = []
        seen_urls: set[str] = set()
        pages = _MAX_PAGES if fetch_all else 1
        truncated = False

        async with self._client() as client:
            for _ in range(pages):
                if url in seen_urls:
                    # Self-referential `next`: stop rather than loop. We were
                    # told more exists but can't safely reach it, so this is
                    # an incomplete answer, not a finished one.
                    truncated = True
                    break
                seen_urls.add(url)

                response = await client.get(url, headers=self._headers(), params=params)
                params = None  # the `next` URL already carries the query

                if response.status_code in (401, 403):
                    raise RuntimeError(
                        "Okta refused the user search. This API token may lack user "
                        "read access across the directory, which is broader than "
                        "reading a single known user."
                    )
                if response.status_code == 400:
                    # Okta answers 400 for a filter it can't parse. Its raw body
                    # is not actionable, and the expression is ours, so name that.
                    raise RuntimeError(
                        "Okta rejected the search expression. This is a bug in how "
                        "the search filter is built, not something a different name "
                        "will fix."
                    )
                response.raise_for_status()

                page = response.json()
                users.extend(page)

                next_url = response.links.get("next", {}).get("url")
                if not next_url:
                    break
                url = next_url
            else:
                # Ran out of page budget with a `next` still outstanding. A
                # hard bound keeps a looping `next` from hanging the CLI, but
                # the answer is incomplete and must say so.
                truncated = True

        if not fetch_all:
            truncated = len(users) >= MAX_SEARCH_RESULTS
        return users, truncated

    async def _call_apps_backend(self, okta_id: str) -> list[dict]:
        if self.mock_mode:
            return self._mock_apps_fixture()

        return await self._fetch_all_pages(
            self._user_url(okta_id, "appLinks"),
            on_denied=(
                "Okta refused the app list request. This API token may lack "
                "application read access, which is granted separately from "
                "user read access."
            ),
        )

    async def _call_factors_backend(self, okta_id: str) -> list[dict]:
        if self.mock_mode:
            return self._mock_factors_fixture()

        return await self._fetch_all_pages(
            self._user_url(okta_id, "factors"),
            on_denied=(
                "Okta refused the authenticator request. This API token may lack "
                "factor read access, which is granted separately from user read "
                "access."
            ),
        )

    async def _call_backend(self, identifier: str) -> dict | None:
        """Return the raw Okta user payload, or None if there's no such user."""
        if self.mock_mode:
            return self._mock_fixture(identifier)

        org_url = self.config.require("OKTA_ORG_URL").rstrip("/")

        # `identifier` is user input. quote(safe="") keeps an email's `@`
        # working while stopping `../` from walking off /api/v1/users.
        url = f"{org_url}/api/v1/users/{quote(identifier, safe='')}"

        async with self._client() as client:
            response = await client.get(url, headers=self._headers())

        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def _client(self) -> httpx.AsyncClient:
        """A timeout-bounded client, so one slow call can't hang a lookup."""
        timeout = float(self.config.get("OKTA_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
        return httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"SSWS {self.config.require('OKTA_API_TOKEN')}",
            "Accept": "application/json",
        }

    def _mock_fixture(self, identifier: str) -> dict:
        return {
            "id": "00uMOCK0000000000000",
            "status": "ACTIVE",
            "created": "2024-01-01T00:00:00.000Z",
            "activated": "2024-01-01T00:05:00.000Z",
            "statusChanged": "2024-01-01T00:05:00.000Z",
            "lastLogin": "2026-01-01T00:00:00.000Z",
            "profile": {
                "firstName": "Mock",
                "lastName": "User",
                "email": f"{identifier}@example.com",
                "login": identifier,
                # Fictional value; the real attribute's type is org-defined
                # and this connector does not care which it is.
                DEFAULT_ACCESS_ATTRIBUTE: False,
            },
        }

    def _mock_devices_fixture(self) -> list[dict]:
        return [
            {
                "id": "guoMOCK00000000000001",
                "managementStatus": "MANAGED",
                "device": {
                    "id": "guoMOCK00000000000001",
                    "status": "ACTIVE",
                    "lastUpdated": "2026-01-01T00:00:00.000Z",
                    "profile": {
                        "displayName": "Mock MacBook Pro",
                        "platform": "MACOS",
                        "manufacturer": "Apple",
                        "model": "MacBookPro18,3",
                        "osVersion": "15.6.0",
                        "serialNumber": "C02MOCK00001",
                    },
                },
            }
        ]

    def _mock_search_fixture(self) -> list[dict]:
        """Three hits including a deactivated one, so the mock demo shows the
        chooser and the case the feature exists for rather than a single hit."""
        return [
            {"id": "00uMOCK1", "status": "ACTIVE",
             "profile": {"login": "dluo", "firstName": "Dennis", "lastName": "Luo",
                         "email": "dluo@example.com"}},
            {"id": "00uMOCK2", "status": "DEPROVISIONED",
             "profile": {"login": "dcarter", "firstName": "Dennis", "lastName": "Carter",
                         "email": "dcarter@example.com"}},
            {"id": "00uMOCK3", "status": "ACTIVE",
             "profile": {"login": "mdennison", "firstName": "Marta", "lastName": "Dennison",
                         "email": "mdennison@example.com"}},
        ]

    def _mock_apps_fixture(self) -> list[dict]:
        return [
            {
                "id": "0oaMOCK00000000000001",
                "label": "Mock Google Workspace",
                "appName": "google",
                "hidden": False,
            },
            {
                "id": "0oaMOCK00000000000002",
                "label": "Mock Slack",
                "appName": "slack",
                "hidden": True,
            },
        ]

    def _mock_factors_fixture(self) -> list[dict]:
        return [
            {
                "id": "opfMOCK00000000000001",
                "factorType": "push",
                "provider": "OKTA",
                "status": "ACTIVE",
                "created": "2025-06-11T08:12:00.000Z",
                "profile": {"name": "Mock iPhone"},
            },
            {
                "id": "opfMOCK00000000000002",
                "factorType": "token:software:totp",
                "provider": "OKTA",
                "status": "ACTIVE",
                "created": "2025-06-11T08:14:00.000Z",
                "profile": {"credentialId": "mock@example.com"},
            },
        ]

    # -- shaping --------------------------------------------------------------

    @staticmethod
    def _to_match(user: dict) -> dict:
        """One search candidate: only what the chooser needs to let someone
        pick, plus the status that usually settles which one they meant."""
        profile = user.get("profile") or {}
        names = [profile.get("firstName"), profile.get("lastName")]
        return {
            "login": profile.get("login"),
            "name": " ".join(part for part in names if part) or None,
            "email": profile.get("email"),
            "status": user.get("status"),
        }

    @staticmethod
    def _to_app(entry: dict) -> dict:
        return {
            "app_id": entry.get("id") or entry.get("appInstanceId"),
            "label": entry.get("label"),
            "app_name": entry.get("appName"),
            # A hidden tile is still an assignment. Recorded rather than
            # filtered, so the CLI can say so instead of silently omitting it.
            "hidden": bool(entry.get("hidden")),
        }

    @staticmethod
    def _to_factor(entry: dict) -> dict:
        factor_type = entry.get("factorType")
        provider = entry.get("provider")
        return {
            "factor_id": entry.get("id"),
            "factor_type": factor_type,
            "provider": provider,
            "label": factor_label(factor_type, provider),
            "detail": factor_detail(entry.get("profile")),
            "status": entry.get("status"),
            "created": entry.get("created"),
        }

    @staticmethod
    def _to_device(entry: dict) -> dict:
        """Normalise one device entry.

        Okta returns a link object wrapping a `device`; tolerate a flat
        device object too, since not every response nests it.
        """
        device = entry.get("device") or entry
        profile = device.get("profile") or {}
        return {
            "device_id": device.get("id") or entry.get("id"),
            "display_name": profile.get("displayName"),
            "platform": profile.get("platform"),
            "manufacturer": profile.get("manufacturer"),
            "model": profile.get("model"),
            "os_version": profile.get("osVersion"),
            "serial_number": profile.get("serialNumber"),
            "status": device.get("status"),
            "management_status": entry.get("managementStatus"),
            "last_updated": device.get("lastUpdated"),
        }


    def _to_result(self, identifier: str, raw: dict) -> ConnectorResult:
        profile = raw.get("profile") or {}
        status = raw.get("status")
        names = [profile.get("firstName"), profile.get("lastName")]
        display_name = " ".join(part for part in names if part) or None

        return ConnectorResult(
            plugin_name=self.name,
            identifier=identifier,
            data={
                "found": True,
                "status": status,
                # Verbatim: whatever Okta returned, uninterpreted. Placed
                # right after `status` so the CLI's ordered walk renders the
                # row directly beneath it.
                "access_blocked": profile.get(self._access_attribute),
                # Derived, but worth carrying: it is the single question
                # offboarding actually asks, and it keeps every consumer
                # (CLI, Stage 7 aggregation, JSON output) from re-deriving
                # which of eight enum values means "deactivated".
                "deactivated": status == _DEACTIVATED_STATUS,
                "login": profile.get("login"),
                "email": profile.get("email"),
                "display_name": display_name,
            },
            # Optional detail goes here rather than growing `data`'s schema.
            properties={
                "okta_id": raw.get("id"),
                "created": raw.get("created"),
                "activated": raw.get("activated"),
                "status_changed": raw.get("statusChanged"),
                "last_login": raw.get("lastLogin"),
            },
            tags=["active" if status in _ACTIVE_STATUSES else "inactive"],
        )

    # -- CLI ------------------------------------------------------------------

    def cli(self) -> typer.Typer:
        """`lookup-cli okta <identifier> [-s] [-d]`.

        Shape: `<service> <person> [what you want]`, with no noun
        subcommands. `okta status jdoe` and `okta devices jdoe` were removed
        on 2026-09-02 because they cannot coexist with `okta jdoe` -- a
        person whose Okta login is literally "status" or "devices" would
        silently resolve to the subcommand instead of being looked up.
        Stages 4-6 follow the same shape.

        Lives here, not in core `cli.py`, so this connector required no edit
        to `src/lookup_cli/`.
        """
        sub_app = typer.Typer(
            help="Okta account lookups.",
            # Required: Click groups stop parsing options once they hit a
            # positional, so without this `okta jdoe -d` fails while
            # `okta -d jdoe` works -- a confusing split for users.
            context_settings={"allow_interspersed_args": True},
        )
        console = Console()

        @sub_app.callback(invoke_without_command=True)
        def okta(
            identifier: str = typer.Argument(..., help="Okta username or email address."),
            status: bool = typer.Option(
                False,
                "--status",
                "-s",
                help="Show account status, including whether the user is deactivated. "
                "This is the default when no other flag is given.",
            ),
            devices: bool = typer.Option(
                False,
                "--devices",
                "-d",
                help="List devices registered to this user in Okta "
                "(Okta Verify / device trust -- not the Jamf inventory).",
            ),
            apps: bool = typer.Option(
                False,
                "--apps",
                "-apps",
                "-a",
                help="List applications assigned to this user, hidden tiles "
                "included. Shows what they can open, not how it was granted.",
            ),
            authenticators: bool = typer.Option(
                False,
                "--authenticators",
                "-authenticators",
                "-u",
                help="List authenticators (MFA factors) this user has enrolled, "
                "including inactive ones.",
            ),
            find: bool = typer.Option(
                False,
                "--find",
                help="Search for a person by name instead of looking one up. "
                "Use when you know their first or surname but not their Okta "
                "username. Long-only and cannot be combined with the section "
                "flags -- it finds a person, it does not describe one.",
            ),
            show_all: bool = typer.Option(
                False,
                "--all",
                help="With --find, show every match instead of the first "
                "screenful, following pagination. Long-only: -a already means "
                "applications.",
            ),
            last_signin: bool = typer.Option(
                False,
                "--last-signin",
                help="Add each device's most recent sign-in, from the Okta System "
                "Log. Opt-in: it costs an extra call to a rate-limited endpoint. "
                "Implies --devices.",
            ),
            since: str = typer.Option(
                "90d",
                "--since",
                help="Window for --last-signin, e.g. 30d or 12h. Okta retains "
                "System Log data for about 90 days, which is the maximum.",
            ),
        ) -> None:
            """Look one person up in Okta, or search for them by name."""
            # --find is a mode, not a section: it answers "who is this
            # person", while every section flag answers "what do you want to
            # see about this person". There is nothing to describe until one
            # has been picked, so the combination is rejected rather than
            # silently dropping the section the user typed -- which is the
            # failure mode this CLI keeps designing out.
            if find:
                conflicting = [
                    name
                    for name, on in (
                        ("--status", status), ("--devices", devices), ("--apps", apps),
                        ("--authenticators", authenticators), ("--last-signin", last_signin),
                    )
                    if on
                ]
                if conflicting:
                    console.print(
                        f"[red]--find cannot be combined with[/red] {', '.join(conflicting)}[red].[/red]\n"
                        "--find locates a person; the section flags describe one already found.\n"
                        f"Find the username first, then: [bold]lookup-cli okta <username> "
                        f"{conflicting[0]}[/bold]"
                    )
                    raise typer.Exit(code=2)
                _print_search(identifier, fetch_all=show_all)
                return

            if show_all:
                # --all modifies the search; there is no search to modify.
                # Accepting it silently would leave someone believing they
                # had asked for something.
                console.print(
                    "[red]--all only applies to --find.[/red]\n"
                    f"Did you mean: [bold]lookup-cli okta --find {identifier} --all[/bold]?"
                )
                raise typer.Exit(code=2)

            # Asking for per-device sign-ins obviously means you want the
            # device table; requiring -d as well would just be pedantry.
            if last_signin:
                devices = True

            window = None
            if last_signin:
                try:
                    window = parse_since(since)
                except ValueError as exc:
                    console.print(f"[red]Invalid --since:[/red] {exc}")
                    raise typer.Exit(code=2)

            # Flags select sections. With none given, status is what people
            # want; `-d` alone means devices only.
            show_status = status or not (devices or apps or authenticators)

            # A section that is the *whole* answer must fail the command, so
            # scripts can trust the exit code. Alongside other sections a dead
            # endpoint degrades its own row instead of discarding good output.
            sole_section = sum((show_status, devices, apps, authenticators)) == 1

            result = asyncio.run(self.fetch(identifier))
            if not result.ok:
                console.print(f"[red]Okta lookup failed:[/red] {result.error}")
                raise typer.Exit(code=1)

            if not result.data.get("found"):
                console.print(f"[yellow]No Okta account found for[/yellow] {identifier}")
                # A hint, not an implicit fallback: --find stays explicit and
                # the miss path stays one API call. Without this, someone who
                # does not know --find exists still hits a dead end -- and not
                # knowing the username is exactly the situation in which you
                # would not know the flag either.
                console.print(
                    f"[dim]Try:[/dim] [bold]lookup-cli okta --find {identifier}[/bold]"
                    "[dim]   to search by name[/dim]"
                )
                return

            okta_id = result.properties.get("okta_id")

            if show_status:
                _print_status(identifier, result)

            if devices:
                _print_devices(identifier, okta_id=okta_id, window=window, primary=sole_section)

            if apps:
                _print_apps(identifier, okta_id=okta_id, primary=sole_section)

            if authenticators:
                _print_authenticators(identifier, okta_id=okta_id, primary=sole_section)

        def _print_search(query: str, fetch_all: bool = False) -> None:
            """Candidate chooser for `--find`.

            Deliberately the same shape as the CAIRO connector's vendor
            chooser: same problem (fuzzy input, several candidates, never
            guess), so an operator learns one idiom rather than two. It is
            reimplemented rather than shared because `plugins/CLAUDE.md`
            forbids importing across plugin packages -- extracting this into
            core is a bigger decision than a connector task should make, and
            is logged in docs/STAGES.md instead.
            """
            result = asyncio.run(self.fetch_search(query, fetch_all=fetch_all))

            if not result.ok:
                console.print(f"[red]Okta search failed:[/red] {result.error}")
                raise typer.Exit(code=1)

            matches = result.data["matches"]
            if not matches:
                console.print(f"[yellow]No Okta user matches[/yellow] {query}")
                console.print(
                    "[dim]Search matches the start of a first name, surname, login or "
                    "email -- so 'dennis' finds Dennis, but 'ennis' finds nobody. "
                    "Multiple words narrow: 'dennis luo' needs both to match.[/dim]"
                )
                return

            shown = matches if fetch_all else matches[:MAX_MATCHES_SHOWN]
            console.print(
                f"[yellow]{result.data['count']} "
                f"{'person' if result.data['count'] == 1 else 'people'} match[/yellow] "
                f"'{query}'[yellow]:[/yellow]"
            )

            table = Table()
            # Login never wraps: it is the value you copy into the next
            # command, and a truncated username is worse than useless.
            table.add_column("username", no_wrap=True)
            table.add_column("name")
            table.add_column("email")
            table.add_column("status", no_wrap=True)
            for match in shown:
                status_value = match["status"] or "UNKNOWN"
                colour, _note = _STATUS_NOTES.get(status_value, ("yellow", ""))
                table.add_row(
                    match["login"] or "-",
                    match["name"] or "-",
                    match["email"] or "-",
                    f"[{colour}]{status_value}[/{colour}]",
                )
            console.print(table)

            if len(matches) > len(shown):
                # Never truncate silently -- a short list reads as "that's all"
                # -- and always name the way out. Telling someone to narrow
                # without mentioning --all repeats the dead end that the
                # --find hint exists to prevent.
                console.print(
                    f"[yellow]{len(matches) - len(shown)} more not shown[/yellow] - "
                    "narrow the search (try adding a surname), or use [bold]--all[/bold]"
                )
            if result.data["truncated"]:
                console.print(
                    "[yellow]Okta may be holding more matches than it returned[/yellow] - "
                    + ("narrow the search to be sure you are seeing everyone."
                       if fetch_all else
                       "narrow the search, or use [bold]--all[/bold] to follow pagination.")
                )

            # Make the two-step flow copy-paste rather than retype.
            example = shown[0]["login"] or "<username>"
            console.print(
                f"[dim]Then:[/dim] [bold]lookup-cli okta {example} -sdau[/bold]"
                "[dim]   (or -s / -d / -a / -u)[/dim]"
            )

        def _print_status(identifier: str, result: ConnectorResult) -> None:
            status_value = result.data.get("status") or "UNKNOWN"
            colour, note = _STATUS_NOTES.get(status_value, ("yellow", ""))
            changed = (result.properties.get("status_changed") or "")[:10]

            suffix = ""
            if note:
                suffix = f" ({note}{' ' + changed if changed else ''})"
            console.print(
                f"[bold]{identifier}[/bold] - [{colour}]{status_value}[/{colour}]{suffix}"
            )

            table = Table(title=f"Okta - {identifier}")
            table.add_column("field")
            table.add_column("value")
            # `found` and `deactivated` are derived and already stated in the
            # line above; repeating them here is noise.
            for key, value in result.data.items():
                if key in ("found", "deactivated"):
                    continue
                if key == "access_blocked":
                    table.add_row(ACCESS_FIELD_LABEL, format_profile_value(value))
                else:
                    table.add_row(key, str(value) if value is not None else "-")
            for key, value in result.properties.items():
                table.add_row(key, str(value) if value is not None else "-")
            console.print(table)

        def _print_apps(identifier: str, okta_id: str | None, primary: bool) -> None:
            result = asyncio.run(self.fetch_apps(identifier, okta_id=okta_id))

            if not result.ok:
                console.print(f"[red]Applications unavailable:[/red] {result.error}")
                if primary:
                    raise typer.Exit(code=1)
                return

            if not result.data.get("found", True):
                console.print(f"[yellow]No Okta account found for[/yellow] {identifier}")
                return

            apps = result.data["apps"]
            if not apps:
                console.print(
                    f"[yellow]No applications assigned in Okta to[/yellow] {identifier}"
                )
                return

            table = Table(title=f"Applications ({result.data['count']}) - {identifier}")
            table.add_column("app")
            table.add_column("type")
            # Named for the API field rather than inverted to "visible": an
            # operator reading the table should not have to flip the sense of
            # the column in their head to match what Okta told us.
            table.add_column("hidden")
            for app in apps:
                table.add_row(
                    app["label"] or "-",
                    app["app_name"] or "-",
                    "yes" if app["hidden"] else "no",
                )
            console.print(table)

        def _print_authenticators(identifier: str, okta_id: str | None, primary: bool) -> None:
            result = asyncio.run(self.fetch_authenticators(identifier, okta_id=okta_id))

            if not result.ok:
                console.print(f"[red]Authenticators unavailable:[/red] {result.error}")
                if primary:
                    raise typer.Exit(code=1)
                return

            if not result.data.get("found", True):
                console.print(f"[yellow]No Okta account found for[/yellow] {identifier}")
                return

            factors = result.data["authenticators"]
            if not factors:
                console.print(f"[yellow]No authenticators enrolled in Okta by[/yellow] {identifier}")
                return

            table = Table(title=f"Authenticators ({result.data['count']}) - {identifier}")
            table.add_column("type")
            # `detail` is the field that says *which* authenticator this is --
            # two Okta Verify pushes are only distinguishable by device name --
            # so it never wraps. The type label gives way instead.
            table.add_column("detail", no_wrap=True)
            table.add_column("status")
            table.add_column("enrolled", no_wrap=True)
            for factor in factors:
                colour = _FACTOR_STATUS_COLOURS.get(factor["status"], "yellow")
                status_value = factor["status"] or "UNKNOWN"
                table.add_row(
                    factor["label"] or "-",
                    factor["detail"] or "-",
                    f"[{colour}]{status_value}[/{colour}]",
                    (factor["created"] or "")[:10] or "-",
                )
            console.print(table)

        def _print_devices(
            identifier: str, okta_id: str | None, primary: bool, window: timedelta | None = None
        ) -> None:
            result = asyncio.run(self.fetch_devices(identifier, okta_id=okta_id))

            if not result.ok:
                console.print(f"[red]Devices unavailable:[/red] {result.error}")
                if primary:
                    raise typer.Exit(code=1)
                return

            if not result.data.get("found", True):
                console.print(f"[yellow]No Okta account found for[/yellow] {identifier}")
                return

            found = result.data["devices"]
            if not found:
                console.print(f"[yellow]No devices registered in Okta for[/yellow] {identifier}")
                return

            # Five columns, not seven: at a stock 80-column terminal rich
            # squeezes seven down until the serial renders as an empty cell.
            # Serial is the field an offboarding operator actually needs, so
            # it never wraps -- the name gives way instead.
            signins: dict[str, str] = {}
            signins_failed = False
            if window is not None:
                # Only the devices we are about to print, so paging can stop
                # as soon as they are all accounted for.
                wanted = {d["device_id"] for d in found if d.get("device_id")}
                signin_result = asyncio.run(
                    self.fetch_device_signins(
                        result.properties.get("okta_id") or okta_id or identifier,
                        since=window,
                        device_ids=wanted or None,
                    )
                )
                if signin_result.ok:
                    signins = signin_result.data["signins"]
                else:
                    # The inventory is a real answer on its own; losing sign-in
                    # times degrades one column rather than discarding it.
                    signins_failed = True
                    console.print(f"[yellow]Sign-in times unavailable:[/yellow] {signin_result.error}")

            table = Table(title=f"Devices ({result.data['count']}) - {identifier}")
            table.add_column("name")
            table.add_column("platform")
            # `model` gives way when the sign-in column is present. Six columns
            # re-create the 80-column squeeze that dropping from seven to five
            # fixed: model truncates to "MacBook..." and status wraps to three
            # lines. Of the two, model is the least actionable -- serial
            # identifies the machine and platform says what it is.
            if window is None:
                table.add_column("model")
            table.add_column("serial", no_wrap=True)
            table.add_column("status")
            if window is not None:
                # The window is in the header, not a footnote: a blank cell
                # means "not in this window", never "never used".
                table.add_column(f"last sign-in ({describe_window(window)})", no_wrap=True)
            for device in found:
                platform = " ".join(
                    part for part in (device["platform"], device["os_version"]) if part
                )
                state = " / ".join(
                    part for part in (device["status"], device["management_status"]) if part
                )
                row = [device["display_name"] or "-", platform or "-"]
                if window is None:
                    row.append(device["model"] or "-")
                row += [device["serial_number"] or "-", state or "-"]
                if window is not None:
                    if signins_failed:
                        row.append("?")
                    else:
                        stamp = signins.get(device.get("device_id") or "")
                        row.append(stamp[:10] if stamp else "-")
                table.add_row(*row)
            console.print(table)

        return sub_app
