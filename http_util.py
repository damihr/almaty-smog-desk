"""Robust JSON GET: retries on rate limits (429), server errors, timeouts and API error bodies."""
import time, requests

def get_json(url, params=None, headers=None, key=None, tries=6, timeout=(10, 60)):
    wait = 5
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code == 200 and (key is None or key in body):
                return body
            reason = body.get("reason") or body.get("detail") or r.text[:200]
            print(f"  [retry {i+1}/{tries}] {url.split('?')[0]} -> HTTP {r.status_code}: {reason}", flush=True)
        except (requests.RequestException, ValueError) as e:
            print(f"  [retry {i+1}/{tries}] {url.split('?')[0]} -> {type(e).__name__}: {e}", flush=True)
        time.sleep(wait); wait = min(wait * 2, 90)
    raise RuntimeError(f"Giving up on {url} after {tries} tries")
