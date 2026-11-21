"""Verify and store Meta publishing access for a Facebook Page and its Instagram account."""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit

import requests

from . import config

GRAPH_URL = "https://graph.facebook.com/v26.0"
TOKEN_PATH = config.DATA_DIR / "meta_token.json"
PUBLISH_PERMISSIONS = {
    "pages_show_list", "pages_read_engagement", "pages_manage_posts",
    "instagram_basic", "instagram_content_publish",
}
_lock = threading.Lock()


class ConnectionError(RuntimeError):
    pass


class PublishUnconfirmed(ConnectionError):
    pass


def _get(path: str, token: str, **params) -> dict:
    try:
        response = requests.get(
            f"{GRAPH_URL}/{path}", params=params, headers={"Authorization": f"Bearer {token}"},
            timeout=20, allow_redirects=False,
        )
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ConnectionError("Could not reach the Meta API. Try again.") from exc
    error = body.get("error", {})
    if error.get("code") == 190:
        raise ConnectionError("Meta access has expired or was revoked. Connect a fresh token.")
    if not response.ok or error:
        raise ConnectionError("Meta did not allow this check. Verify the app permissions and selected accounts.")
    return body


def _read() -> dict | None:
    try:
        return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status() -> dict:
    saved = _read() or {}
    return {"connected": bool(saved.get("access_token")), "facebook": saved.get("facebook"),
            "instagram": saved.get("instagram"),
            "instagram_reels": bool(saved.get("instagram_access_token"))}


def _verify_page_token(token: str, page_id: str, instagram_id: str) -> None:
    if _get("me", token, fields="id").get("id") != page_id:
        raise ConnectionError("Meta returned a token for a different Facebook Page.")
    _get(f"{instagram_id}/content_publishing_limit", token, fields="quota_usage,config")


def connect(user_token: str, page_id: str, instagram_username: str) -> dict:
    granted = {
        entry["permission"] for entry in _get("me/permissions", user_token).get("data", [])
        if entry.get("status") == "granted"
    }
    if missing := PUBLISH_PERMISSIONS - granted:
        raise ConnectionError("Grant these Meta permissions before connecting: " + ", ".join(sorted(missing)))
    page = _get(
        page_id, user_token, fields="id,name,access_token,instagram_business_account{id,username}",
    )
    instagram = page.get("instagram_business_account") or {}
    page_token = page.get("access_token")
    if page.get("id") != page_id or not page_token:
        raise ConnectionError("Meta did not provide publishing access to that Facebook Page.")
    if not instagram.get("id") or instagram.get("username", "").lower() != instagram_username.lstrip("@").lower():
        raise ConnectionError("The Facebook Page is linked to a different Instagram account.")
    _verify_page_token(page_token, page_id, instagram["id"])
    _get(f"{instagram['id']}/content_publishing_limit", user_token, fields="quota_usage,config")
    facebook = {"id": page_id, "name": page["name"]}
    instagram = {"id": instagram["id"], "username": instagram["username"]}
    with _lock:
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = TOKEN_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"access_token": page_token, "instagram_access_token": user_token,
                                   "facebook": facebook, "instagram": instagram}),
                       encoding="utf-8")
        tmp.replace(TOKEN_PATH)
    return status()


def verify() -> dict:
    saved = _read()
    if not saved or not saved.get("access_token"):
        raise ConnectionError("Connect Meta before testing the connection.")
    _verify_page_token(saved["access_token"], saved["facebook"]["id"], saved["instagram"]["id"])
    if instagram_token := saved.get("instagram_access_token"):
        _get(f"{saved['instagram']['id']}/content_publishing_limit", instagram_token, fields="quota_usage,config")
    return status()


def _post(path: str, token: str, *, final: bool = False, **fields) -> dict:
    try:
        reply = requests.post(
            f"{GRAPH_URL}/{path}", data=fields, headers={"Authorization": f"Bearer {token}"},
            timeout=35, allow_redirects=False,
        )
        result = reply.json()
    except (requests.RequestException, ValueError):
        if final:
            raise PublishUnconfirmed("Meta did not confirm the publish request. Check Published in Meta Business Suite.") from None
        raise ConnectionError("Meta could not start the upload. Try again.") from None
    if final and reply.status_code >= 500:
        raise PublishUnconfirmed("Meta did not confirm the publish request. Check Published in Meta Business Suite.")
    if not reply.ok or result.get("error") or result.get("success") is False:
        error = result.get("error") or {}
        if error.get("code") == 190:
            raise ConnectionError("Meta access has expired or was revoked. Reconnect the account in Integrations.")
        message = str(error.get("message") or "Meta rejected this request.").replace(token, "[redacted]")
        raise ConnectionError(message[:300])
    return result


def _upload_video(url: str, video: Path, token: str) -> None:
    destination = urlsplit(url)
    if destination.scheme != "https" or destination.netloc != "rupload.facebook.com":
        raise ConnectionError("Meta returned an unexpected upload destination.")
    try:
        with video.open("rb") as stream:
            response = requests.post(url, data=stream, headers={
                "Authorization": f"OAuth {token}",
                "offset": "0",
                "file_size": str(video.stat().st_size),
            }, timeout=(20, 180), allow_redirects=False)
        result = response.json()
    except (requests.RequestException, ValueError, OSError):
        raise ConnectionError("The video could not be uploaded to Meta. Try again.") from None
    if not response.ok or not result.get("success"):
        error = result.get("error") or result.get("debug_info") or {}
        message = error.get("message") if isinstance(error, dict) else error
        detail = str(message or "Check the clip format or try again.").replace(token, "[redacted]")
        raise ConnectionError(f"Meta rejected the video upload. {detail[:210]}")


def _prepare_video(source: Path, output: Path, platform: str) -> None:
    # Keep the rendered H.264 stream. A short remux normalizes the audio and MP4 metadata.
    duration = ["-t", "89.8"] if platform == "facebook" else []
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source),
             *duration, "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy",
             "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "128k",
             "-movflags", "+faststart", "-use_editlist", "0", str(output)],
            check=True, capture_output=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        raise ConnectionError("Arsuclip could not prepare this clip for Reels.") from None


def publish_reel(
    platform: str, account_id: str, path: Path, title: str, caption: str,
    progress: Callable[[str], None],
) -> dict:
    account = _read() or {}
    token = account.get("access_token")
    identity = account.get(platform) or {}
    if not token or identity.get("id") != account_id:
        raise ConnectionError("The connected Meta account changed. Check Integrations and try again.")
    if platform == "instagram":
        token = account.get("instagram_access_token")
        if not token:
            raise ConnectionError("Reconnect Facebook & Instagram in Integrations to enable direct Instagram posting.")
    progress("Preparing video")
    with tempfile.TemporaryDirectory(dir=config.DATA_DIR, prefix="reel-") as folder:
        video = Path(folder) / "reel.mp4"
        _prepare_video(path, video, platform)
        progress("Uploading video")
        if platform == "facebook":
            edge = f"{account_id}/video_reels"
            session = _post(edge, token, upload_phase="start")
            video_id = session["video_id"]
            _upload_video(session["upload_url"], video, token)
            progress("Confirming publication")
            _post(edge, token, final=True, upload_phase="finish", video_id=video_id,
                  video_state="PUBLISHED", title=title, description=caption, is_ai_generated="true")
            try:
                for _ in range(12):
                    status = _get(video_id, token, fields="status,permalink_url")
                    phase = status.get("status", {}).get("publishing_phase", {})
                    if phase.get("publish_status") == "published":
                        return {"id": video_id, "url": urljoin("https://www.facebook.com",
                                status.get("permalink_url") or f"/reel/{video_id}/")}
                    if phase.get("status") == "error":
                        raise ConnectionError("Facebook reported a publishing error. Check the Page before retrying.")
                    time.sleep(10)
            except ConnectionError:
                pass
            raise PublishUnconfirmed("Facebook accepted the upload. Check Published in Meta Business Suite for its final status.")

        session = _post(f"{account_id}/media", token, upload_type="resumable", media_type="REELS",
                        share_to_feed="true", caption=caption, is_ai_generated="true")
        container_id = session["id"]
        _upload_video(session["uri"], video, token)
        progress("Instagram is processing the video")
        for _ in range(6):
            state = _get(container_id, token, fields="status_code").get("status_code")
            if state == "FINISHED":
                break
            if state in ("ERROR", "EXPIRED"):
                raise ConnectionError(f"Instagram could not process this upload ({state.lower()}).")
            time.sleep(20)
        else:
            raise ConnectionError("Instagram did not finish processing the upload. Try again.")
        progress("Confirming publication")
        posted = _post(f"{account_id}/media_publish", token, final=True, creation_id=container_id)
        if not (video_id := posted.get("id")):
            raise PublishUnconfirmed("Instagram did not return a Reel ID. Check Published in Meta Business Suite.")
        result = {"id": video_id}
        try:
            item = _get(video_id, token, fields="permalink")
        except ConnectionError:
            return result
        if item.get("permalink"):
            result["url"] = item["permalink"]
        return result
