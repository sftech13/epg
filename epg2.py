#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import time
import secrets
import signal
import fcntl
import atexit
import re
import contextlib
import sqlite3
import datetime as _dt
import requests
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

# ---------------- Global Defaults ----------------
CONFIG_FILE = str(Path(__file__).parent / "epg_config.json")
DB_FILE = "/home/sftech13/git/epg/zap2it.db"
_LOCK_FILE = "/tmp/epg_standalone.run.lock"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
]

BASE_URL = "https://tvlistings.gracenote.com/api/grid"
COUNTRY_3 = {"US": "USA", "CA": "CAN", "UK": "GBR", "GB": "GBR"}
COUNTRY_2 = {"USA": "us", "CAN": "ca", "GBR": "uk", "United States": "us", "Canada": "ca", "United Kingdom": "uk"}

# ---------------- Logging ----------------
def setup_logging(verbosity, log_file: str = None):
    """Setup logging with appropriate level and optional file output."""
    if isinstance(verbosity, str):
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'WARN': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        level = level_map.get(verbosity.upper(), logging.INFO)
    else:
        verbosity = int(verbosity)
        if verbosity == 0:
            level = logging.WARNING
        elif verbosity == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file {log_file}: {e}", file=sys.stderr)
    
    logging.basicConfig(level=level, handlers=handlers, force=True)

# ---------------- Single Instance Lock ----------------
def _release_lock():
    """Release EPG lock file safely."""
    if os.path.exists(_LOCK_FILE):
        try:
            os.remove(_LOCK_FILE)
            logging.debug(f"Released lock file: {_LOCK_FILE}")
        except OSError as e:
            logging.warning(f"Could not remove lock file {_LOCK_FILE}: {e}")

@contextlib.contextmanager
def _single_instance_guard():
    """Ensure only one instance of the script runs at a time using flock."""
    fd = None
    try:
        os.makedirs('/tmp', exist_ok=True)
        fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode('utf-8'))
        
        atexit.register(_release_lock)
        signal.signal(signal.SIGTERM, lambda *a, **k: (_release_lock(), os._exit(0)))
        signal.signal(signal.SIGINT,  lambda *a, **k: (_release_lock(), os._exit(0)))
        
        yield
        
    except BlockingIOError:
        logging.error("Another instance is already running. Exiting.")
        raise SystemExit(1)
    except Exception as e:
        logging.error(f"Failed to acquire lock: {e}")
        raise SystemExit(1)
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except Exception:
                pass
        _release_lock()

# ---------------- API helpers ----------------
_CDN_SUBDOMAIN = "zap2it"

def _set_cdn_subdomain(subdomain: str):
    """Set the CDN subdomain for image URLs."""
    global _CDN_SUBDOMAIN
    _CDN_SUBDOMAIN = subdomain

def _ua():
    """Return a random user agent using secrets for better randomness."""
    return USER_AGENTS[secrets.randbelow(len(USER_AGENTS))]

def _is_ota(lineup_id: str) -> bool:
    """Check if lineup is Over-The-Air broadcast."""
    s = (lineup_id or "").upper()
    return "OTA" in s or "LOCALBROADCAST" in s or s.endswith("-DEFAULT")

def _headend_from_lineup(lineup_id: str) -> str:
    """Extract headend ID from lineup ID."""
    if _is_ota(lineup_id):
        return "lineupId"
    m = re.match(r"^[A-Z]{3}-([^-]+)-", lineup_id or "")
    return m.group(1) if m else "lineup"

def _device_from_lineup(lineup_id: str) -> str:
    """Extract device identifier from lineup ID."""
    s = (lineup_id or "").upper().strip()
    if _is_ota(s) or s.endswith("-DEFAULT"):
        return "-"
    m = re.search(r"-([A-Z])$", s)
    return m.group(1) if m else "-"

def _fix_thumbnail_url(thumbnail: str) -> str:
    """Fix incomplete thumbnail URLs from GraceNote API to use proper tmsimg.com CDN."""
    if not thumbnail:
        return ""
    
    has_extension = any(thumbnail.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])
    
    if thumbnail.startswith("http://") or thumbnail.startswith("https://"):
        return thumbnail if has_extension else f"{thumbnail}.jpg"
    
    if thumbnail.startswith("//"):
        url = f"https:{thumbnail}"
        return url if has_extension else f"{url}.jpg"
    
    if thumbnail.startswith("assets/"):
        return f"https://{_CDN_SUBDOMAIN}.tmsimg.com/{thumbnail}.jpg"
    else:
        return f"https://{_CDN_SUBDOMAIN}.tmsimg.com/assets/{thumbnail}.jpg"

def _build_url(lineup_id: str, headend_id: str, country: str, postal: str, 
               time_sec: int, chunk_hours: int, is_ota: bool) -> str:
    """Build API URL with proper parameters."""
    device = _device_from_lineup(lineup_id)
    user_id = secrets.token_hex(4)

    params = [
        ("lineupId", lineup_id),
        ("timespan", str(chunk_hours)),
        ("headendId", headend_id),
        ("country", country),
        ("device", device),
        ("isOverride", "true"),
        ("time", str(time_sec)),
        ("pref", "16,128"),
        ("userId", user_id),
        ("aid", "chi"),
        ("languagecode", "en-us"),
    ]

    if is_ota and postal:
        params.insert(6, ("postalCode", str(postal)))
    else:
        params.insert(6, ("postalCode", "-"))

    qs = "&".join(f"{k}={v}" for k, v in params if v not in (None, ""))
    return f"{BASE_URL}?{qs}"

# ---------------- Fetch Logic ----------------
def fetch_grid(country: str, lineup_id_input: str, postal: str, timespan: int = 72, 
               delay_seconds: int = 0, max_retries: int = 3, 
               session: requests.Session = None) -> Tuple[List[Dict], Dict]:
    """
    Fetch EPG data for a given lineup and return channels/events with enriched metadata.
    Returns: (channels_list, stats_dict)
    """
    c3 = COUNTRY_3.get(country.upper(), country.upper())
    lineup_api = lineup_id_input
    headend_api = _headend_from_lineup(lineup_id_input)
    total_hours = int(timespan)
    chunk_hours = 6
    
    sess = session or requests.Session()
    channels_map = {}
    base_time = int(time.time())
    offsets = list(range(0, total_hours, chunk_hours))
    is_ota = _is_ota(lineup_id_input)
    
    stats = {
        "chunks_fetched": 0,
        "chunks_failed": 0,
        "total_chunks": len(offsets),
        "channels_found": 0,
        "events_found": 0,
        "is_ota": is_ota
    }

    for idx, offset in enumerate(offsets):
        t = base_time + offset * 3600
        url = _build_url(lineup_api, headend_api, c3, postal or "", t, chunk_hours, is_ota=is_ota)

        logging.debug(f"[{lineup_id_input}] Fetching chunk {idx+1}/{len(offsets)}")

        for attempt in range(1, max_retries + 1):
            headers = {"User-Agent": _ua()}
            try:
                r = sess.get(url, headers=headers, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    for ch in data.get("channels", []) or []:
                        cid = str(ch.get("channelId"))
                        if cid not in channels_map:
                            channels_map[cid] = {
                                "stationId": ch.get("stationId") or ch.get("channelId"),
                                "channelId": ch.get("channelId"),
                                "callSign": ch.get("callSign") or ch.get("name"),
                                "channelNo": ch.get("channelNo") or ch.get("channel"),
                                "affiliateName": ch.get("affiliateName"),
                                "thumbnail": _fix_thumbnail_url(ch.get("thumbnail")),
                                "events": [],
                                "isOTA": is_ota
                            }
                        base = channels_map[cid]
                        for ev in ch.get("events", []) or []:
                            prog = ev.get("program", {}) or {}
                            enriched_event = {
                                "startTime": ev.get("startTime"),
                                "endTime": ev.get("endTime"),
                                "title": prog.get("title") or ev.get("title"),
                                "description": prog.get("shortDesc") or ev.get("description"),
                                "seasonTitle": prog.get("seasonTitle") or prog.get("episodeTitle"),
                                "showType": prog.get("showType"),
                                "runTime": prog.get("runTime"),
                                "isNew": ev.get("isNew") or prog.get("isNew") or prog.get("new") or ("New" in (ev.get("flag") or [])),
                                "seasonPremiere": prog.get("seasonPremiere"),
                                "seasonFinale": prog.get("seasonFinale"),
                                "genres": prog.get("genres") or [],
                                "language": prog.get("language"),
                                "thumbnail": _fix_thumbnail_url(prog.get("thumbnail") or ev.get("thumbnail")),
                                "preferredImage": _fix_thumbnail_url((prog.get("preferredImage") or {}).get("uri")),
                                "seasonNum": prog.get("seasonNum") or prog.get("season"),
                                "episodeNum": prog.get("episodeNum") or prog.get("episode"),
                                "seriesId": prog.get("seriesId") or ev.get("seriesId"),
                                "programId": prog.get("programId") or prog.get("id"),
                                "cast": prog.get("cast") or [],
                                "crew": prog.get("crew") or [],
                                "ratings": _extract_rating(prog.get("ratings")),
                            }
                            base["events"].append(enriched_event)
                    stats["chunks_fetched"] += 1
                    break

                elif r.status_code == 400:
                    logging.error(f"[{lineup_id_input}] Bad request (likely invalid headend or postalCode)")
                    stats["chunks_failed"] += 1
                    break

                elif r.status_code in (429,) or 500 <= r.status_code < 600:
                    wait = min(60, 2 ** (attempt - 1)) + secrets.randbelow(1000) / 1000
                    logging.warning(f"[{lineup_id_input}] Retry {attempt}/{max_retries} in {wait:.1f}s (status {r.status_code})")
                    time.sleep(wait)
                else:
                    logging.error(f"[{lineup_id_input}] HTTP {r.status_code} for chunk {idx+1}")
                    stats["chunks_failed"] += 1
                    break

            except Exception as e:
                if attempt == max_retries:
                    logging.error(f"[{lineup_id_input}] Failed after {max_retries} attempts: {e}")
                    stats["chunks_failed"] += 1
                    break
                wait = min(30, 2 ** (attempt - 1)) + secrets.randbelow(1000) / 1000
                logging.warning(f"[{lineup_id_input}] Network error: {e} � retrying in {wait:.1f}s")
                time.sleep(wait)

        if delay_seconds > 0 and idx < len(offsets) - 1:
            time.sleep(delay_seconds)

    for ch in channels_map.values():
        ch["events"] = _deduplicate_events_enhanced(ch["events"])
    
    channels = list(channels_map.values())
    channels.sort(key=lambda c: (str(c.get("callSign") or ""), str(c.get("channelNo") or "")))
    
    stats["channels_found"] = len(channels)
    stats["events_found"] = sum(len(ch.get("events", [])) for ch in channels)
    
    return channels, stats

def _extract_rating(ratings_data):
    """Extract rating from various formats returned by GraceNote."""
    if not ratings_data:
        return None
    
    if isinstance(ratings_data, list) and len(ratings_data) > 0:
        rating = ratings_data[0]
        if isinstance(rating, dict):
            return {
                "system": rating.get("body", ""),
                "value": rating.get("code", "")
            }
    
    if isinstance(ratings_data, dict):
        return {
            "system": ratings_data.get("body", ""),
            "value": ratings_data.get("code", "")
        }
    
    return None

def _deduplicate_events_enhanced(events: List[Dict]) -> List[Dict]:
    """Remove duplicate events and prefer ones with richer metadata."""
    seen = {}
    
    def _metadata_score(ev):
        score = 0
        if ev.get("description"): score += 3
        if ev.get("thumbnail") or ev.get("preferredImage"): score += 2
        if ev.get("seasonNum") is not None: score += 2
        if ev.get("episodeNum") is not None: score += 2
        if ev.get("seasonTitle"): score += 1
        if ev.get("genres"): score += 1
        if ev.get("ratings"): score += 1
        if ev.get("cast"): score += 1
        if ev.get("runTime"): score += 1
        return score
    
    for ev in sorted(events, key=lambda e: e.get("startTime") or ""):
        key = (ev.get("startTime"), ev.get("title"))
        
        if key not in seen:
            seen[key] = ev
        else:
            if _metadata_score(ev) > _metadata_score(seen[key]):
                seen[key] = ev
    
    return list(seen.values())

def _normalize_callsign(callsign: str) -> str:
    """Normalize call sign for better merging."""
    if not callsign:
        return ""
    cs = callsign.upper().strip()
    cs = re.sub(r'\b(HD|SD|DT|TV|PLUS|FEED)\b', '', cs)
    cs = re.sub(r'\(\d+\)', '', cs)
    cs = re.sub(r'[^A-Z0-9 ]', '', cs)
    cs = re.sub(r'\s+', ' ', cs).strip()
    return cs

def merge_duplicate_channels(all_channels: List[Dict]) -> List[Dict]:
    """Merge channels with the same normalized call sign or affiliate name."""
    channels_by_key = defaultdict(list)

    for ch in all_channels:
        call_sign = _normalize_callsign(ch.get("callSign")) \
                    or _normalize_callsign(ch.get("affiliateName")) \
                    or str(ch.get("channelNo") or "").strip()
        channels_by_key[call_sign].append(ch)

    merged_channels = []

    for key, channel_group in channels_by_key.items():
        if len(channel_group) == 1:
            merged_channels.append(channel_group[0])
            continue

        logging.debug(f"Merging {len(channel_group)} instances of channel group '{key}'")

        base = max(channel_group, key=lambda c: (
            bool(c.get("thumbnail")),
            bool(c.get("affiliateName")),
            len(c.get("events", []))
        ))

        station_id = base.get("stationId")
        if not station_id:
            for ch in channel_group:
                if ch.get("stationId"):
                    station_id = ch["stationId"]
                    break
        base["stationId"] = station_id

        all_events = []
        for ch in channel_group:
            all_events.extend(ch.get("events", []))

        base["events"] = _deduplicate_events_enhanced(all_events)
        merged_channels.append(base)

    merged_channels.sort(key=lambda c: (
        _normalize_callsign(c.get("callSign")) or "",
        int(c.get("channelNo") or 0) if str(c.get("channelNo", "")).replace(".", "").isdigit() else 0
    ))

    if len(merged_channels) != len(all_channels):
        logging.debug(f"After merging: {len(merged_channels)} unique channels (was {len(all_channels)})")

    return merged_channels

def get_country_suffix(lineup_id: str, db_path: str = DB_FILE) -> str:
    """Get country suffix for channel IDs based on lineup."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT country, country_name FROM channels_by_country WHERE lineupid = ? LIMIT 1", (lineup_id,))
        result = cur.fetchone()
        conn.close()
        
        if result:
            country_code = result[0] or ""
            country_name = result[1] or ""
            suffix = COUNTRY_2.get(country_code) or COUNTRY_2.get(country_name)
            if suffix:
                return suffix
            if len(country_code) >= 2:
                return country_code[:2].lower()
        
        if lineup_id.startswith("USA"):
            return "us"
        elif lineup_id.startswith("CAN"):
            return "ca"
        elif lineup_id.startswith("GBR") or lineup_id.startswith("UK"):
            return "uk"
        
        return "xx"
        
    except Exception as e:
        logging.warning(f"Could not determine country suffix for {lineup_id}: {e}")
        return "xx"

# ---------------- Time & XML helpers ----------------
def _zap_iso_to_dt(s):
    """Convert various time formats to datetime object."""
    if not s:
        return None
    try:
        st = str(s)
        if st.endswith('Z'):
            return _dt.datetime.fromisoformat(st[:-1]).replace(tzinfo=_dt.timezone.utc)
        if re.fullmatch(r"\d{10}", st):
            return _dt.datetime.fromtimestamp(int(st), tz=_dt.timezone.utc)
        return _dt.datetime.fromisoformat(st.replace('Z',''))
    except Exception:
        return None

def _xmltv_time(dtobj):
    """Convert datetime to XMLTV format."""
    if not dtobj:
        return ""
    if dtobj.tzinfo is None:
        dtobj = dtobj.replace(tzinfo=_dt.timezone.utc)
    return dtobj.strftime("%Y%m%d%H%M%S %z")

def _validate_event(ev: Dict, ch_id: str) -> bool:
    """Validate event has required fields and logical times."""
    start_dt = _zap_iso_to_dt(ev.get("startTime"))
    end_dt = _zap_iso_to_dt(ev.get("endTime"))
    
    if not start_dt or not end_dt:
        logging.debug(f"Event missing start/end time on channel {ch_id}")
        return False
    
    if start_dt >= end_dt:
        logging.debug(f"Event has invalid time range on channel {ch_id}: {start_dt} >= {end_dt}")
        return False
    
    return True

def write_xmltv(channels: List[Dict], out_path: Path, include_thumbnails: bool = True, 
                country_suffix_map: Dict[str, str] = None):
    """Write channels and events to XMLTV format with country-based channel IDs."""
    tv = ET.Element("tv")
    tv.set("generator-info-name", "GraceNote EPG Fetcher Enhanced")
    tv.set("generator-info-url", "https://github.com")
    
    channels = sorted(channels, key=lambda c: str(c.get("callSign") or "").lower())

    for ch in channels:
        call_sign = ch.get("callSign") or ch.get("affiliateName") or str(ch.get("channelNo") or "")
        call_sign = re.sub(r'[^A-Za-z0-9]', '', call_sign)
        
        country_suffix = "xx"
        if country_suffix_map:
            station_id = str(ch.get("stationId") or ch.get("channelId") or "")
            country_suffix = country_suffix_map.get(station_id, "xx")
        
        channel_id = f"{call_sign}.{country_suffix}".lower()
        
        ch_el = ET.SubElement(tv, "channel", {"id": channel_id})

        for val in [
            ch.get("callSign"),
            ch.get("affiliateName"),
            f"{ch.get('callSign')} {ch.get('affiliateName')}" if ch.get("callSign") and ch.get("affiliateName") else None
        ]:
            if val:
                ET.SubElement(ch_el, "display-name").text = str(val)

        thumb = ch.get("thumbnail")
        if include_thumbnails and thumb:
            if thumb.startswith("http") and "." in thumb:
                ET.SubElement(ch_el, "icon", {"src": str(thumb)})

    valid_events = 0
    invalid_events = 0
    new_episodes = 0
    reruns = 0
    
    for ch in channels:
        call_sign = ch.get("callSign") or ch.get("affiliateName") or str(ch.get("channelNo") or "")
        call_sign = re.sub(r'[^A-Za-z0-9]', '', call_sign)
        
        country_suffix = "xx"
        if country_suffix_map:
            station_id = str(ch.get("stationId") or ch.get("channelId") or "")
            country_suffix = country_suffix_map.get(station_id, "xx")
        
        channel_id = f"{call_sign}.{country_suffix}".lower()
        events = sorted(ch.get("events", []), key=lambda e: e.get("startTime") or "")
        
        for ev in events:
            if not _validate_event(ev, channel_id):
                invalid_events += 1
                continue
                
            start_dt = _zap_iso_to_dt(ev.get("startTime"))
            end_dt = _zap_iso_to_dt(ev.get("endTime"))
            
            prog_el = ET.SubElement(tv, "programme", {
                "start": _xmltv_time(start_dt),
                "stop": _xmltv_time(end_dt),
                "channel": channel_id,
            })

            title = ev.get("title")
            if title:
                ET.SubElement(prog_el, "title", {"lang": "en"}).text = str(title)

            subtitle = ev.get("seasonTitle")
            if subtitle:
                ET.SubElement(prog_el, "sub-title", {"lang": "en"}).text = str(subtitle)

            desc = ev.get("description")
            if desc:
                ET.SubElement(prog_el, "desc", {"lang": "en"}).text = str(desc)

            show_type = ev.get("showType")
            if show_type:
                ET.SubElement(prog_el, "category", {"lang": "en"}).text = str(show_type)
            
            genres = ev.get("genres") or []
            if isinstance(genres, list):
                for genre in genres[:5]:
                    if genre:
                        ET.SubElement(prog_el, "category", {"lang": "en"}).text = str(genre)

            language = ev.get("language")
            if language:
                ET.SubElement(prog_el, "language").text = str(language)

            runtime = ev.get("runTime")
            if runtime:
                try:
                    minutes = int(runtime) if int(runtime) < 500 else int(runtime) // 60
                    ET.SubElement(prog_el, "length", {"units": "minutes"}).text = str(minutes)
                except (ValueError, TypeError):
                    pass

            season_num = ev.get("seasonNum")
            episode_num = ev.get("episodeNum")
            if season_num is not None or episode_num is not None:
                s = int(season_num) - 1 if season_num else 0
                e = int(episode_num) - 1 if episode_num else 0
                xmltv_ns = f"{s}.{e}."
                ET.SubElement(prog_el, "episode-num", {"system": "xmltv_ns"}).text = xmltv_ns

            series_id = ev.get("seriesId") or ev.get("programId")
            if series_id:
                ET.SubElement(prog_el, "episode-num", {"system": "dd_progid"}).text = str(series_id)

            prog_thumb = ev.get("preferredImage") or ev.get("thumbnail")
            if prog_thumb and include_thumbnails:
                if prog_thumb.startswith("http") and "." in prog_thumb:
                    ET.SubElement(prog_el, "icon", {"src": str(prog_thumb)})

            ratings = ev.get("ratings")
            if ratings:
                rating_el = ET.SubElement(prog_el, "rating")
                if ratings.get("system"):
                    rating_el.set("system", str(ratings["system"]))
                if ratings.get("value"):
                    ET.SubElement(rating_el, "value").text = str(ratings["value"])

            cast_list = ev.get("cast") or []
            crew_list = ev.get("crew") or []
            if cast_list or crew_list:
                credits_el = ET.SubElement(prog_el, "credits")
                
                for actor in cast_list[:10]:
                    if isinstance(actor, dict):
                        name = actor.get("name")
                        role = actor.get("role")
                        if name:
                            actor_el = ET.SubElement(credits_el, "actor")
                            actor_el.text = str(name)
                            if role:
                                actor_el.set("role", str(role))
                    elif isinstance(actor, str):
                        ET.SubElement(credits_el, "actor").text = str(actor)
                
                for crew_member in crew_list[:10]:
                    if isinstance(crew_member, dict):
                        name = crew_member.get("name")
                        role = crew_member.get("role", "").lower()
                        if name and role:
                            if "director" in role:
                                ET.SubElement(credits_el, "director").text = str(name)
                            elif "writer" in role or "author" in role:
                                ET.SubElement(credits_el, "writer").text = str(name)
                            elif "producer" in role:
                                ET.SubElement(credits_el, "producer").text = str(name)

            is_new = ev.get("isNew")
            if is_new is True:
                ET.SubElement(prog_el, "new")
                new_episodes += 1
            elif is_new is False:
                ET.SubElement(prog_el, "previously-shown")
                reruns += 1

            if ev.get("seasonPremiere"):
                ET.SubElement(prog_el, "premiere")
            if ev.get("seasonFinale"):
                ET.SubElement(prog_el, "last-chance")
            
            valid_events += 1

    if invalid_events > 0:
        logging.warning(f"Skipped {invalid_events} invalid events during XML generation")
    
    logging.info(f"Writing {valid_events} valid events to XML")
    logging.info(f"  New episodes: {new_episodes}")
    logging.info(f"  Reruns: {reruns}")
    logging.info(f"  Unknown status: {valid_events - new_episodes - reruns}")

    tree = ET.ElementTree(tv)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)

# ---------------- DB Lookup ----------------
def get_lineups_from_column(column: str, value: str, max_lineups: int = None, 
                           db_path: str = DB_FILE, ota_only: bool = False) -> List[str]:
    """Query database for lineups matching a column value."""
    if not Path(db_path).exists():
        logging.error(f"Database file {db_path} not found")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        kw = f"%{value.lower()}%"
        
        query = f"SELECT DISTINCT lineupid FROM channels_by_country WHERE lower({column}) LIKE ?"
        if ota_only:
            query += " AND (lineupid LIKE '%OTA%' OR lineupid LIKE '%LOCALBROADCAST%' OR lineupid LIKE '%-DEFAULT')"
        
        cur.execute(query, (kw,))
        results = [row[0] for row in cur.fetchall()]
        conn.close()
        
        if max_lineups and len(results) > max_lineups:
            results = results[:max_lineups]
        
        logging.debug(f"Found {len(results)} lineup IDs for {column}='{value}'" + (" (OTA only)" if ota_only else ""))
        return results
    except Exception as e:
        logging.error(f"Database query failed: {e}")
        return []

def get_lineups_from_keyword(keyword: str, max_lineups: int = None, 
                            db_path: str = DB_FILE, ota_only: bool = False) -> List[str]:
    """Query database for lineups matching keyword across multiple columns."""
    if not Path(db_path).exists():
        logging.error(f"Database file {db_path} not found")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        kw = f"%{keyword.lower()}%"
        
        query = """
            SELECT DISTINCT lineupid FROM channels_by_country
            WHERE lower(lineup_name) LIKE ?
               OR lower(country_name) LIKE ?
               OR lower(country) LIKE ?
               OR lower(station_name) LIKE ?
               OR lower(lineupid) LIKE ?
        """
        
        if ota_only:
            query += " AND (lineupid LIKE '%OTA%' OR lineupid LIKE '%LOCALBROADCAST%' OR lineupid LIKE '%-DEFAULT')"
        
        cur.execute(query, (kw, kw, kw, kw, kw))
        results = [row[0] for row in cur.fetchall()]
        conn.close()
        
        if max_lineups and len(results) > max_lineups:
            results = results[:max_lineups]
        
        logging.debug(f"Found {len(results)} lineup IDs for keyword='{keyword}'" + (" (OTA only)" if ota_only else ""))
        return results
    except Exception as e:
        logging.error(f"Database query failed: {e}")
        return []

def get_ota_lineups_by_country(country: str, db_path: str = DB_FILE, max_lineups: int = None) -> List[str]:
    """Get all OTA (local broadcast) lineups for a specific country."""
    if not Path(db_path).exists():
        logging.error(f"Database file {db_path} not found")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        query = """
            SELECT DISTINCT lineupid FROM channels_by_country
            WHERE (lower(country) = ? OR lower(country_name) = ?)
            AND (lineupid LIKE '%OTA%' OR lineupid LIKE '%LOCALBROADCAST%' OR lineupid LIKE '%-DEFAULT')
        """
        
        country_lower = country.lower()
        cur.execute(query, (country_lower, country_lower))
        results = [row[0] for row in cur.fetchall()]
        conn.close()
        
        if max_lineups and len(results) > max_lineups:
            results = results[:max_lineups]
        
        logging.info(f"Found {len(results)} OTA lineups for country '{country}'")
        return results
    except Exception as e:
        logging.error(f"Database query failed: {e}")
        return []

def filter_broken_lineups(lineup_ids: List[str]) -> List[str]:
    """Remove known broken/test lineups."""
    bad_patterns = ["xumotv", "sandbox", "test", "dummy", "dtvnow", "amzpv", 
                    "youtube", "gnstr", "imdbtv"]
    result = []
    for lid in lineup_ids:
        lid_lower = lid.lower()
        if any(bad in lid_lower for bad in bad_patterns):
            logging.warning(f"Filtered out broken lineup: {lid}")
            continue
        result.append(lid)
    return result

# ---------------- Config Loader ----------------
def load_config(args=None) -> Dict:
    """Load configuration from file, environment variables, and command-line arguments."""
    config = {
        "EPG_LINEUP_ID": "",
        "EPG_LOOKUP_MODE": "",
        "EPG_LOOKUP_VALUE": "",
        "EPG_COUNTRY": "USA",
        "EPG_ZIP": "",
        "EPG_OUTPUT": "EPG.xml",
        "EPG_TIMESPAN_DAYS": 1,
        "EPG_DELAY": 0,
        "EPG_VERBOSE": 1,
        "EPG_LOG_LEVEL": "INFO",
        "EPG_DB_PATH": DB_FILE,
        "EPG_THUMBNAILS": True,
        "EPG_MAX_LINEUPS": None,
        "EPG_RETRY_COUNT": 3,
        "EPG_LOG_FILE": None,
        "EPG_PARALLEL": False,
        "EPG_MAX_WORKERS": 3,
        "EPG_OTA_ONLY": False,
        "EPG_CDN_SUBDOMAIN": "zap2it",
    }
    
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                file_cfg = json.load(f)
            config.update({k.upper(): v for k, v in file_cfg.items()})
            logging.debug(f"Loaded configuration from {CONFIG_FILE}")
        except Exception as e:
            logging.warning(f"Failed to read {CONFIG_FILE}: {e}")
    
    for key in ["EPG_TIMESPAN_DAYS", "EPG_VERBOSE", "EPG_DELAY", "EPG_MAX_LINEUPS", 
                "EPG_RETRY_COUNT", "EPG_MAX_WORKERS"]:
        if config.get(key) is not None and not isinstance(config[key], int):
            try:
                config[key] = int(config[key])
            except (ValueError, TypeError):
                logging.warning(f"Invalid integer for {key}: {config[key]}, using default")
                config[key] = None
    
    for key in config.keys():
        if key in os.environ:
            val = os.environ[key]
            if key in ["EPG_TIMESPAN_DAYS", "EPG_VERBOSE", "EPG_DELAY", "EPG_MAX_LINEUPS", 
                      "EPG_RETRY_COUNT", "EPG_MAX_WORKERS"]:
                try:
                    val = int(val)
                except ValueError:
                    logging.warning(f"Invalid integer for {key}: {val}")
                    continue
            elif key in ["EPG_THUMBNAILS", "EPG_PARALLEL", "EPG_OTA_ONLY"]:
                val = val.lower() in ("true", "1", "yes", "on")
            config[key] = val

    if args:
        if args.lineup_id:
            config["EPG_LINEUP_ID"] = args.lineup_id
        if args.country:
            config["EPG_COUNTRY"] = args.country
        if args.zip:
            config["EPG_ZIP"] = args.zip
        if args.output:
            config["EPG_OUTPUT"] = args.output
        if args.days:
            config["EPG_TIMESPAN_DAYS"] = args.days
        if args.delay:
            config["EPG_DELAY"] = args.delay
        if args.log_level:
            config["EPG_LOG_LEVEL"] = args.log_level
        if args.log_file:
            config["EPG_LOG_FILE"] = args.log_file
        if args.db_path:
            config["EPG_DB_PATH"] = args.db_path
        if args.max_lineups:
            config["EPG_MAX_LINEUPS"] = args.max_lineups
        if args.retry_count:
            config["EPG_RETRY_COUNT"] = args.retry_count
        if args.parallel:
            config["EPG_PARALLEL"] = True
        if args.max_workers:
            config["EPG_MAX_WORKERS"] = args.max_workers
        if args.no_thumbnails:
            config["EPG_THUMBNAILS"] = False
        if args.ota_only:
            config["EPG_OTA_ONLY"] = True
        if args.lookup_mode:
            config["EPG_LOOKUP_MODE"] = args.lookup_mode
        if args.lookup_value:
            config["EPG_LOOKUP_VALUE"] = args.lookup_value
        if args.cdn_subdomain:
            config["EPG_CDN_SUBDOMAIN"] = args.cdn_subdomain

    _set_cdn_subdomain(config.get("EPG_CDN_SUBDOMAIN", "zap2it"))

    days = int(config.get("EPG_TIMESPAN_DAYS", 1))
    config["EPG_TIMESPAN"] = min(max(days, 1), 7) * 24

    if not config["EPG_LINEUP_ID"] and config["EPG_LOOKUP_MODE"] and config["EPG_LOOKUP_VALUE"]:
        max_lineups = config.get("EPG_MAX_LINEUPS")
        mode = config["EPG_LOOKUP_MODE"].lower()
        raw_val = config["EPG_LOOKUP_VALUE"]
        lookup_values = [v.strip() for v in raw_val.split(",") if v.strip()]
        db_path = config.get("EPG_DB_PATH", DB_FILE)
        ota_only = config.get("EPG_OTA_ONLY", False)
        all_lineups = []

        logging.info(f"Looking up lineups with mode='{mode}', values={lookup_values}" + (" (OTA only)" if ota_only else ""))

        for val in lookup_values:
            if mode == "country_name":
                all_lineups.extend(get_lineups_from_column("country_name", val, max_lineups, db_path, ota_only))
            elif mode == "country":
                all_lineups.extend(get_lineups_from_column("country", val, max_lineups, db_path, ota_only))
            elif mode == "station_name":
                all_lineups.extend(get_lineups_from_column("station_name", val, max_lineups, db_path, ota_only))
            elif mode == "keyword":
                all_lineups.extend(get_lineups_from_keyword(val, max_lineups, db_path, ota_only))
            elif mode == "ota":
                all_lineups.extend(get_ota_lineups_by_country(val, db_path, max_lineups))
            else:
                logging.error(f"Unsupported lookup mode: {mode}")

        filtered_lineups = filter_broken_lineups(all_lineups)
        unique_lineups = list(dict.fromkeys(filtered_lineups))

        if max_lineups and len(unique_lineups) > max_lineups:
            unique_lineups = unique_lineups[:max_lineups]

        if unique_lineups:
            config["EPG_LINEUP_ID"] = ",".join(unique_lineups)
            logging.info(f"Resolved to {len(unique_lineups)} lineup IDs")
        else:
            logging.warning(f"No lineups found for mode='{mode}', values={lookup_values}")

    return config

# ---------------- Parallel Fetcher ----------------
def fetch_lineup_wrapper(args: Tuple) -> Tuple[str, List[Dict], Dict]:
    """Wrapper for parallel execution."""
    lineup, country, postal, timespan, delay, retry_count = args
    try:
        channels, stats = fetch_grid(
            country=country,
            lineup_id_input=lineup,
            postal=postal,
            timespan=timespan,
            delay_seconds=delay,
            max_retries=retry_count
        )
        return lineup, channels, stats
    except Exception as e:
        logging.error(f"Failed to fetch lineup {lineup}: {e}")
        return lineup, [], {"error": str(e)}

# ---------------- Runner ----------------
def run_epg(config: Dict):
    """Main EPG fetching logic."""
    lineup_ids = [lid.strip() for lid in str(config["EPG_LINEUP_ID"]).split(",") if lid.strip()]
    
    if not lineup_ids:
        logging.error("No lineup IDs found.")
        logging.error(f"  Lookup mode: {config.get('EPG_LOOKUP_MODE')}")
        logging.error(f"  Lookup value: {config.get('EPG_LOOKUP_VALUE')}")
        logging.error("  Check your EPG_LOOKUP_MODE/EPG_LOOKUP_VALUE or set EPG_LINEUP_ID directly.")
        sys.exit(1)

    country = config["EPG_COUNTRY"]
    postal = config["EPG_ZIP"]
    timespan = int(config["EPG_TIMESPAN"])
    delay = int(config["EPG_DELAY"])
    retry_count = int(config.get("EPG_RETRY_COUNT", 3))
    output_file = Path(config["EPG_OUTPUT"])
    parallel = config.get("EPG_PARALLEL", False)
    max_workers = int(config.get("EPG_MAX_WORKERS", 3))
    db_path = config.get("EPG_DB_PATH", DB_FILE)

    logging.info(f"Starting EPG fetch for {len(lineup_ids)} lineup(s)")
    logging.info(f"  Timespan: {timespan} hours")
    logging.info(f"  Parallel: {parallel} (workers: {max_workers if parallel else 'N/A'})")
    
    all_channels = []
    all_stats = {}
    country_suffix_map = {}
    
    if parallel and len(lineup_ids) > 1:
        args_list = [
            (lid, country, postal, timespan, delay, retry_count) 
            for lid in lineup_ids
        ]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_lineup_wrapper, args): args[0] for args in args_list}
            
            for future in as_completed(futures):
                lineup = futures[future]
                try:
                    lineup_id, channels, stats = future.result()
                    all_channels.extend(channels)
                    all_stats[lineup_id] = stats
                    
                    suffix = get_country_suffix(lineup_id, db_path)
                    for ch in channels:
                        station_id = str(ch.get("stationId") or ch.get("channelId") or "")
                        if station_id:
                            country_suffix_map[station_id] = suffix
                    
                    logging.info(f"  [{lineup_id}] {stats.get('channels_found', 0)} channels, "
                               f"{stats.get('events_found', 0)} events")
                except Exception as e:
                    logging.error(f"  [{lineup}] Failed: {e}")
                    all_stats[lineup] = {"error": str(e)}
    else:
        for idx, lineup in enumerate(lineup_ids, 1):
            logging.info(f"Fetching lineup {idx}/{len(lineup_ids)}: {lineup}")
            try:
                channels, stats = fetch_grid(
                    country=country,
                    lineup_id_input=lineup,
                    postal=postal,
                    timespan=timespan,
                    delay_seconds=delay,
                    max_retries=retry_count
                )
                all_channels.extend(channels)
                all_stats[lineup] = stats
                
                suffix = get_country_suffix(lineup, db_path)
                for ch in channels:
                    station_id = str(ch.get("stationId") or ch.get("channelId") or "")
                    if station_id:
                        country_suffix_map[station_id] = suffix
                
                logging.info(f"  [{lineup}] {stats.get('channels_found', 0)} channels, "
                           f"{stats.get('events_found', 0)} events")
            except Exception as e:
                logging.error(f"  [{lineup}] Failed: {e}")
                all_stats[lineup] = {"error": str(e)}

    successful = sum(1 for s in all_stats.values() if "error" not in s)
    failed = len(all_stats) - successful
    
    logging.info(f"Fetch complete: {successful} succeeded, {failed} failed")
    logging.info(f"  Raw channels collected: {len(all_channels)}")
    
    logging.info("Merging duplicate channels across lineups...")
    all_channels = merge_duplicate_channels(all_channels)
    
    total_channels = len(all_channels)
    total_events = sum(len(ch.get("events", [])) for ch in all_channels)
    
    logging.info(f"  Final unique channels: {total_channels}")
    logging.info(f"  Total events after deduplication: {total_events}")

    if not all_channels:
        logging.error("No channels retrieved. Exiting.")
        sys.exit(1)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_xmltv(all_channels, output_file, 
                   include_thumbnails=config.get("EPG_THUMBNAILS", True),
                   country_suffix_map=country_suffix_map)
        logging.info(f"EPG successfully written to {output_file}")
    except Exception as e:
        logging.error(f"Failed to write XMLTV: {e}")
        sys.exit(1)

# ---------------- Command-Line Interface ----------------
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced EPG Fetcher with GraceNote API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch specific lineup
  %(prog)s --lineup-id USA-CA90210-X --country US --zip 90210

  # Fetch all US OTA (local broadcast) channels
  %(prog)s --lookup-mode ota --lookup-value "United States" --ota-only

  # Fetch by country name (OTA only)
  %(prog)s --lookup-mode country_name --lookup-value "United States" --ota-only

  # Fetch multiple lineups in parallel
  %(prog)s --lineup-id USA-CA90210-X,USA-NY10001-X --parallel --max-workers 5

  # Fetch with verbose logging
  %(prog)s --lineup-id USA-CA90210-X --log-level DEBUG --log-file epg.log
        """
    )
    
    lineup_group = parser.add_argument_group('Lineup Selection')
    lineup_group.add_argument('--lineup-id', type=str, 
                             help='Comma-separated lineup IDs (e.g., USA-CA90210-X)')
    lineup_group.add_argument('--lookup-mode', type=str, 
                             choices=['country', 'country_name', 'station_name', 'keyword', 'ota'],
                             help='Database lookup mode')
    lineup_group.add_argument('--lookup-value', type=str,
                             help='Value to search for (comma-separated for multiple)')
    lineup_group.add_argument('--ota-only', action='store_true',
                             help='Filter to only OTA (Over-The-Air) local broadcast channels')
    
    location_group = parser.add_argument_group('Location')
    location_group.add_argument('--country', type=str,
                               help='Country code (US, CA, UK, etc.)')
    location_group.add_argument('--zip', type=str,
                               help='Postal/ZIP code')
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output', '-o', type=str,
                             help='Output XMLTV file path (default: EPG.xml)')
    output_group.add_argument('--no-thumbnails', action='store_true',
                             help='Exclude thumbnail/icon URLs from output')
    
    fetch_group = parser.add_argument_group('Fetching Options')
    fetch_group.add_argument('--days', type=int,
                            help='Number of days to fetch (1-7, default: 1)')
    fetch_group.add_argument('--delay', type=int,
                            help='Delay in seconds between API requests')
    fetch_group.add_argument('--retry-count', type=int,
                            help='Number of retries for failed requests (default: 3)')
    fetch_group.add_argument('--max-lineups', type=int,
                            help='Maximum number of lineups to fetch from database lookup')
    
    parallel_group = parser.add_argument_group('Parallel Execution')
    parallel_group.add_argument('--parallel', action='store_true',
                               help='Fetch multiple lineups in parallel')
    parallel_group.add_argument('--max-workers', type=int,
                               help='Maximum parallel workers (default: 3)')
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--log-level', type=str,
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Logging level')
    log_group.add_argument('--log-file', type=str,
                          help='Log file path (logs to console if not specified)')
    
    db_group = parser.add_argument_group('Database')
    db_group.add_argument('--db-path', type=str,
                         help=f'Path to SQLite database (default: {DB_FILE})')
    
    cdn_group = parser.add_argument_group('CDN Options')
    cdn_group.add_argument('--cdn-subdomain', type=str,
                          help='TMS Image CDN subdomain (default: zap2it)')
    
    return parser.parse_args()

# ---------------- Main ----------------
if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args)
    
    log_level = config.get("EPG_LOG_LEVEL") or config.get("EPG_VERBOSE", 1)
    setup_logging(log_level, config.get("EPG_LOG_FILE"))
    
    try:
        with _single_instance_guard():
            run_epg(config)
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)