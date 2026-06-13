"""Delivery-ambiguity classification for native-forwarder event POSTs.

The claude-native and codex-native forwarders mirror transcript items
into AP as ``external_conversation_item`` POSTs. The server persists
those with a random primary key and does NOT dedupe them — producers
are responsible for not re-posting items they have already sent. That
makes a blind retry after a failed POST unsafe: if the server committed
the item and published ``session.input.consumed`` but the response was
lost, a retry appends a second copy and the web UI renders a duplicate
bubble. The native tmux pane is unaffected, which is why the
duplicate is web-only.

:func:`post_may_have_been_delivered` is the shared classifier both
forwarders use to decide whether a failed POST is safe to retry.
"""

from __future__ import annotations

import httpx

# Transport failures proving a POST never reached the server (no bytes
# sent) — safe to retry. See :func:`post_may_have_been_delivered`.
_DELIVERY_SAFE_RETRY_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


def post_may_have_been_delivered(exc: httpx.HTTPError) -> bool:
    """
    Return whether a failed AP POST may have been delivered AND
    committed by the server despite the error — making a blind retry
    unsafe for non-idempotent events.

    - ``HTTPStatusError``: the server responded with a status. The
      events route returns 2xx only after the item is appended and the
      consume event is published, so any non-2xx means the item was not
      committed (4xx rejects at parse time; a 5xx fails before/at the
      append). No duplicate risk → safe to retry, so ``False``.
    - Connection-establishment / pool-acquire failures
      (:data:`_DELIVERY_SAFE_RETRY_ERRORS`): no bytes were sent → not
      delivered → safe to retry, so ``False``.
    - Any other transport error (read/write timeout, read/write error,
      remote protocol error): the request was sent and we never saw a
      response, so the server may have processed it → ambiguous →
      ``True``.

    :param exc: HTTP exception raised while posting an AP event.
    :returns: ``True`` when a retry could duplicate a server-committed
        item; ``False`` when retrying is safe.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return False
    if isinstance(exc, _DELIVERY_SAFE_RETRY_ERRORS):
        return False
    return True
