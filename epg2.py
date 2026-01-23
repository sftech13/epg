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
VERSION = "2.3.0"
CONFIG_FILE = str(Path(__file__).parent / "epg_config.json")
DB_FILE = str(Path(__file__).parent / "zap2it.db")
_LOCK_FILE = "/tmp/epg_standalone.run.lock"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
]

BASE_URL = "https://tvlistings.gracenote.com/api/grid"
PROVIDER_URL = "https://tvlistings.gracenote.com/gapzap_webapi/api/Providers/getPostalCodeProviders"
COUNTRY_3 = {"US": "USA", "CA": "CAN", "UK": "GBR", "GB": "GBR"}
COUNTRY_2 = {"USA": "us", "CAN": "ca", "GBR": "uk", "United States": "us", "Canada": "ca", "United Kingdom": "uk"}

# Gracenote supported countries with sample postal codes for database updates
GRACENOTE_COUNTRIES = {
    # North America - CONFIRMED WORKING
    'USA': {'name': 'United States', 'sample_postals': ['10001', '90210', '60601', '77001', '94102', '30301', '33101', '98101', '85001', '02101']},
    'CAN': {'name': 'Canada', 'sample_postals': ['M5H2N2', 'V6B1A1', 'H2X1Y7', 'T2P3N4', 'R3C0C1']},

    # Latin America - CONFIRMED WORKING
    'ARG': {'name': 'Argentina', 'sample_postals': ['C1000', 'B1900', 'X5000']},
    'BRA': {'name': 'Brazil', 'sample_postals': ['01000-000', '22041-080', '30130-000']},
    'CHL': {'name': 'Chile', 'sample_postals': ['8320000', '2520000', '4030000']},
    'COL': {'name': 'Colombia', 'sample_postals': ['110111', '050001', '760001']},
    'CRI': {'name': 'Costa Rica', 'sample_postals': ['10101', '20101', '30101']},
    'ECU': {'name': 'Ecuador', 'sample_postals': ['170150', '090112', '010101']},
    'GTM': {'name': 'Guatemala', 'sample_postals': ['01001', '01010', '09001']},
    'HND': {'name': 'Honduras', 'sample_postals': ['11101', '21101', '31301']},
    'MEX': {'name': 'Mexico', 'sample_postals': ['06600', '44100', '64000']},
    'PAN': {'name': 'Panama', 'sample_postals': ['0801', '0401', '0601']},
    'PER': {'name': 'Peru', 'sample_postals': ['15001', '04001', '13001']},
    'URY': {'name': 'Uruguay', 'sample_postals': ['11000', '20000', '60000']},
    'VEN': {'name': 'Venezuela', 'sample_postals': ['1010', '2001', '3001']},

    # Caribbean
    'BHS': {'name': 'Bahamas', 'sample_postals': ['BS']},
    'BRB': {'name': 'Barbados', 'sample_postals': ['BB14001']},
    'BMU': {'name': 'Bermuda', 'sample_postals': ['CR01', 'FL01']},
    'CYM': {'name': 'Cayman Islands', 'sample_postals': ['KY1-1001']},
    'DOM': {'name': 'Dominican Republic', 'sample_postals': ['10101', '10201']},
    'JAM': {'name': 'Jamaica', 'sample_postals': ['JMAAW01', 'JMAKN01']},
    'PRI': {'name': 'Puerto Rico', 'sample_postals': ['00901', '00601', '00921']},
    'TTO': {'name': 'Trinidad and Tobago', 'sample_postals': ['TT']},

    # Europe - MAY HAVE LIMITED SUPPORT
    'AUT': {'name': 'Austria', 'sample_postals': ['1010', '5020', '8010']},
    'BEL': {'name': 'Belgium', 'sample_postals': ['1000', '2000', '9000']},
    'CHE': {'name': 'Switzerland', 'sample_postals': ['8001', '3011', '1201']},
    'DEU': {'name': 'Germany', 'sample_postals': ['10115', '80331', '20095']},
    'DNK': {'name': 'Denmark', 'sample_postals': ['1050', '8000', '5000']},
    'ESP': {'name': 'Spain', 'sample_postals': ['28001', '08001', '46001']},
    'FIN': {'name': 'Finland', 'sample_postals': ['00100', '33100', '20100']},
    'FRA': {'name': 'France', 'sample_postals': ['75001', '69001', '13001']},
    'GBR': {'name': 'United Kingdom', 'sample_postals': ['SW1A', 'M1', 'B1', 'G1', 'EH1']},
    'IRL': {'name': 'Ireland', 'sample_postals': ['D01', 'D02', 'T12']},
    'ITA': {'name': 'Italy', 'sample_postals': ['00118', '20121', '80121']},
    'NLD': {'name': 'Netherlands', 'sample_postals': ['1012', '3011', '6211']},
    'NOR': {'name': 'Norway', 'sample_postals': ['0150', '5003', '7010']},
    'POL': {'name': 'Poland', 'sample_postals': ['00-001', '30-001', '80-001']},
    'SWE': {'name': 'Sweden', 'sample_postals': ['11120', '41101', '21135']},

    # Asia-Pacific
    'AUS': {'name': 'Australia', 'sample_postals': ['2000', '3000', '4000', '5000', '6000']},
    'NZL': {'name': 'New Zealand', 'sample_postals': ['1010', '6011', '8011']},
}

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
    """Check if lineup is Over-The-Air broadcast.

    Only USA OTA lineups (containing 'OTA' or 'LOCALBROADCAST') are true OTA.
    European/other -DEFAULT lineups are cable/satellite, not OTA.
    """
    s = (lineup_id or "").upper()
    # Only USA lineups with OTA or LOCALBROADCAST are real OTA
    # European -DEFAULT lineups (like DEU-1000143-DEFAULT) are cable, not OTA
    if "OTA" in s or "LOCALBROADCAST" in s:
        return True
    # The special lineupId-DEFAULT format is used for API calls to OTA
    if s == "LINEUPID-DEFAULT" or "-LINEUPID-DEFAULT" in s:
        return True
    return False

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
    from urllib.parse import quote

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

    # URL-encode parameter values (especially for postal codes with spaces)
    qs = "&".join(f"{quote(k, safe='')}={quote(str(v), safe='')}" for k, v in params if v not in (None, ""))
    return f"{BASE_URL}?{qs}"

# ---------------- Fetch Logic ----------------
def _api_lineup_and_headend(country: str, lineup_id: str) -> Tuple[str, str]:
    """
    Transform lineup ID and headend for API calls.
    For OTA lineups: lineupId becomes {COUNTRY}-lineupId-DEFAULT, headendId becomes "lineupId"
    For non-OTA: lineup stays as-is, headend is extracted from the lineup ID.
    """
    c3 = COUNTRY_3.get(country.upper(), country.upper())
    if _is_ota(lineup_id):
        return f"{c3}-lineupId-DEFAULT", "lineupId"
    return lineup_id, _headend_from_lineup(lineup_id)


def fetch_grid(country: str, lineup_id_input: str, postal: str, timespan: int = 72,
               delay_seconds: int = 0, max_retries: int = 3,
               session: requests.Session = None) -> Tuple[List[Dict], Dict]:
    """
    Fetch EPG data for a given lineup and return channels/events with enriched metadata.
    Returns: (channels_list, stats_dict)
    """
    c3 = COUNTRY_3.get(country.upper(), country.upper())
    lineup_api, headend_api = _api_lineup_and_headend(c3, lineup_id_input)
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
                            # Rating can be at event level (string like "TV-14") or prog level
                            event_rating = ev.get("rating")  # Direct rating string
                            prog_rating = _extract_rating(prog.get("ratings"))
                            if event_rating and not prog_rating:
                                prog_rating = {"system": "VCHIP", "value": event_rating}

                            # Duration: event.duration is minutes, prog.runTime may also exist
                            duration = ev.get("duration") or prog.get("runTime")

                            # Tags provide audio/video info like Stereo, CC, HD
                            tags = ev.get("tags") or []

                            enriched_event = {
                                "startTime": ev.get("startTime"),
                                "endTime": ev.get("endTime"),
                                "title": prog.get("title") or ev.get("title"),
                                "description": prog.get("shortDesc") or ev.get("description"),
                                "seasonTitle": prog.get("seasonTitle") or prog.get("episodeTitle"),
                                "showType": prog.get("showType"),
                                "runTime": duration,
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
                                "programId": prog.get("programId") or prog.get("id") or prog.get("tmsId"),
                                "tmsId": prog.get("tmsId") or prog.get("id"),
                                "releaseYear": prog.get("releaseYear"),
                                "cast": prog.get("cast") or [],
                                "crew": prog.get("crew") or [],
                                "ratings": prog_rating,
                                "stereo": "Stereo" in tags,
                                "cc": "CC" in tags,
                                "hd": "HD" in tags or "HDTV" in tags,
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

    def _safe_channel_sort_key(ch):
        """Sort channels safely, handling subchannels like '7.1' correctly."""
        call = _normalize_callsign(ch.get("callSign")) or ""
        num_str = str(ch.get("channelNo") or "").strip()
        try:
            # Gracefully handle floats (e.g. "7.1") or missing values
            num = float(num_str) if re.match(r"^\d+(\.\d+)?$", num_str) else 0.0
        except (ValueError, TypeError):
            num = 0.0
        return (call, num)

    merged_channels.sort(key=_safe_channel_sort_key)


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

            # Release year (primarily for movies)
            release_year = ev.get("releaseYear")
            if release_year:
                ET.SubElement(prog_el, "date").text = str(release_year)

            # Audio/Video quality indicators
            if ev.get("stereo") or ev.get("cc") or ev.get("hd"):
                if ev.get("stereo"):
                    audio_el = ET.SubElement(prog_el, "audio")
                    ET.SubElement(audio_el, "stereo").text = "stereo"
                if ev.get("hd"):
                    video_el = ET.SubElement(prog_el, "video")
                    ET.SubElement(video_el, "quality").text = "HDTV"
                if ev.get("cc"):
                    subtitles_el = ET.SubElement(prog_el, "subtitles", {"type": "teletext"})

            valid_events += 1

    if invalid_events > 0:
        logging.warning(f"Skipped {invalid_events} invalid events during XML generation")
    
    logging.info(f"Writing {valid_events} valid events to XML")
    logging.info(f"  New episodes: {new_episodes}")
    logging.info(f"  Reruns: {reruns}")
    #logging.info(f"  Unknown status: {valid_events - new_episodes - reruns}")

    tree = ET.ElementTree(tv)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    # Write to gzip if path ends with .gz, otherwise plain XML
    out_str = str(out_path)
    if out_str.endswith('.gz'):
        import gzip
        with gzip.open(out_str, 'wb') as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
    else:
        tree.write(out_str, encoding="utf-8", xml_declaration=True)

# ---------------- External EPG Sources ----------------
# Supported external EPG providers
EXTERNAL_EPG_SOURCES = {
    # globetvapp/epg - European countries
    "globetvapp": {
        "base_url": "https://raw.githubusercontent.com/globetvapp/epg/main",
        "countries": {
            "austria": "Austria/austria1.xml",
            "belgium": "Belgium/belgium1.xml",
            "bulgaria": "Bulgaria/bulgaria1.xml",
            "croatia": "Croatia/croatia1.xml",
            "czech": "Czechrepublic/czechrepublic1.xml",
            "denmark": "Denmark/denmark1.xml",
            "finland": "Finland/finland1.xml",
            "france": "France/france1.xml",
            "germany": "Germany/germany1.xml",
            "greece": "Greece/greece1.xml",
            "hungary": "Hungary/hungary1.xml",
            "ireland": "Ireland/ireland1.xml",
            "italy": "Italy/italy1.xml",
            "netherlands": "Netherlands/netherlands1.xml",
            "norway": "Norway/norway1.xml",
            "poland": "Poland/poland1.xml",
            "portugal": "Portugal/portugal1.xml",
            "romania": "Romania/romania1.xml",
            "russia": "Russia/russia1.xml",
            "serbia": "Serbia/serbia1.xml",
            "slovakia": "Slovakia/slovakia1.xml",
            "slovenia": "Slovenia/slovenia1.xml",
            "spain": "Spain/spain1.xml",
            "sweden": "Sweden/sweden1.xml",
            "switzerland": "Switzerland/switzerland1.xml",
            "turkey": "Turkey/turkey1.xml",
            "ukraine": "Ukraine/ukraine1.xml",
            "uk": "Unitedkingdom/unitedkingdom1.xml",
            "united kingdom": "Unitedkingdom/unitedkingdom1.xml",
        }
    },
    # i.mjh.nz - Streaming services EPG (via raw GitHub)
    # Source: https://github.com/matthuisman/i.mjh.nz
    "mjh": {
        "base_url": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master",
        "services": {
            # Pluto TV - available in many regions
            "pluto_us": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/us.xml.gz",
            "pluto_uk": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/gb.xml.gz",
            "pluto_ca": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/ca.xml.gz",
            "pluto_de": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/de.xml.gz",
            "pluto_fr": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/fr.xml.gz",
            "pluto_es": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/es.xml.gz",
            "pluto_it": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/it.xml.gz",
            "pluto_mx": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/mx.xml.gz",
            "pluto_br": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/br.xml.gz",
            "pluto_all": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/PlutoTV/all.xml.gz",
            # Samsung TV+ - available in several regions
            "samsung_us": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/SamsungTVPlus/us.xml.gz",
            "samsung_uk": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/SamsungTVPlus/gb.xml.gz",
            "samsung_ca": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/SamsungTVPlus/ca.xml.gz",
            "samsung_de": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/SamsungTVPlus/de.xml.gz",
            "samsung_fr": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/SamsungTVPlus/fr.xml.gz",
            "samsung_es": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/SamsungTVPlus/es.xml.gz",
            "samsung_it": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/SamsungTVPlus/it.xml.gz",
            "samsung_all": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/SamsungTVPlus/all.xml.gz",
            # Plex - available in several regions
            "plex_us": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/Plex/us.xml.gz",
            "plex_uk": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/Plex/gb.xml.gz",
            "plex_ca": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/Plex/ca.xml.gz",
            "plex_mx": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/Plex/mx.xml.gz",
            "plex_all": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/Plex/all.xml.gz",
            # Stirr - US only (all.xml is the only file)
            "stirr": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/Stirr/all.xml.gz",
            # Roku - all regions combined (all.xml is the only file)
            "roku": "https://raw.githubusercontent.com/matthuisman/i.mjh.nz/master/Roku/all.xml.gz",
        }
    }
}


def fetch_external_epg(url: str, max_retries: int = 3) -> Optional[ET.Element]:
    """Download XMLTV from external URL and return parsed XML root element."""
    import gzip
    from io import BytesIO

    for attempt in range(1, max_retries + 1):
        try:
            headers = {"User-Agent": _ua()}
            logging.debug(f"Fetching external EPG: {url} (attempt {attempt}/{max_retries})")

            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()

            content = response.content

            # Handle gzipped content
            if url.endswith('.gz') or response.headers.get('Content-Encoding') == 'gzip':
                try:
                    content = gzip.decompress(content)
                except gzip.BadGzipFile:
                    pass  # Not actually gzipped, use as-is

            # Parse XML
            root = ET.fromstring(content)

            if root.tag != 'tv':
                logging.warning(f"Unexpected root element '{root.tag}' in {url}")
                return None

            channel_count = len(root.findall('.//channel'))
            programme_count = len(root.findall('.//programme'))
            logging.info(f"Downloaded {url}: {channel_count} channels, {programme_count} programmes")

            return root

        except requests.RequestException as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logging.warning(f"Failed to fetch {url}: {e}, retrying in {wait}s")
                time.sleep(wait)
            else:
                logging.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                return None
        except ET.ParseError as e:
            logging.error(f"Failed to parse XML from {url}: {e}")
            return None

    return None


def merge_xmltv_elements(roots: List[ET.Element]) -> ET.Element:
    """Merge multiple XMLTV root elements into one, deduplicating channels."""
    merged = ET.Element("tv")
    merged.set("generator-info-name", "EPG Fetcher External Merge")

    seen_channels = {}  # id -> channel element
    all_programmes = []

    for root in roots:
        if root is None:
            continue

        # Collect channels (dedupe by id)
        for channel in root.findall('.//channel'):
            ch_id = channel.get('id')
            if ch_id and ch_id not in seen_channels:
                seen_channels[ch_id] = channel

        # Collect all programmes
        for programme in root.findall('.//programme'):
            all_programmes.append(programme)

    # Add channels sorted by display-name
    def get_display_name(ch):
        dn = ch.find('display-name')
        return (dn.text or '').lower() if dn is not None else ''

    for ch_id, channel in sorted(seen_channels.items(), key=lambda x: get_display_name(x[1])):
        merged.append(channel)

    # Add programmes sorted by channel then start time
    all_programmes.sort(key=lambda p: (p.get('channel', ''), p.get('start', '')))
    for prog in all_programmes:
        # Only include if channel exists
        if prog.get('channel') in seen_channels:
            merged.append(prog)

    return merged


def resolve_external_urls(profile: Dict) -> List[str]:
    """Resolve external EPG URLs from profile configuration."""
    urls = []
    source = profile.get("source", "").lower()

    # Direct URLs
    if profile.get("urls"):
        raw_urls = profile["urls"]
        if isinstance(raw_urls, str):
            urls.extend([u.strip() for u in raw_urls.split(",") if u.strip()])
        elif isinstance(raw_urls, list):
            urls.extend(raw_urls)

    # globetvapp countries
    if source == "globetvapp" or profile.get("globetvapp_countries"):
        countries_raw = profile.get("globetvapp_countries") or profile.get("countries", "")
        if isinstance(countries_raw, str):
            countries = [c.strip().lower() for c in countries_raw.split(",") if c.strip()]
        else:
            countries = [c.lower() for c in countries_raw]

        base_url = EXTERNAL_EPG_SOURCES["globetvapp"]["base_url"]
        country_map = EXTERNAL_EPG_SOURCES["globetvapp"]["countries"]

        for country in countries:
            if country in country_map:
                urls.append(f"{base_url}/{country_map[country]}")
            else:
                logging.warning(f"Unknown globetvapp country: {country}")

    # i.mjh.nz streaming services
    if source == "mjh" or profile.get("mjh_services"):
        services_raw = profile.get("mjh_services") or profile.get("services", "")
        if isinstance(services_raw, str):
            services = [s.strip().lower() for s in services_raw.split(",") if s.strip()]
        else:
            services = [s.lower() for s in services_raw]

        service_map = EXTERNAL_EPG_SOURCES["mjh"]["services"]

        for service in services:
            if service in service_map:
                urls.append(service_map[service])
            else:
                logging.warning(f"Unknown mjh service: {service}")

    return urls


def fetch_and_merge_external(profile: Dict, output_path: Path, max_retries: int = 3) -> Dict:
    """Fetch external EPG sources and merge into single output file."""
    import gzip

    urls = resolve_external_urls(profile)

    if not urls:
        return {"status": "error", "message": "No URLs resolved for external profile"}

    logging.info(f"Fetching {len(urls)} external EPG source(s)")

    roots = []
    stats = {"urls_fetched": 0, "urls_failed": 0, "total_urls": len(urls)}

    for url in urls:
        root = fetch_external_epg(url, max_retries)
        if root is not None:
            roots.append(root)
            stats["urls_fetched"] += 1
        else:
            stats["urls_failed"] += 1

    if not roots:
        return {"status": "error", "message": "All external fetches failed", "stats": stats}

    # Merge all fetched EPGs
    logging.info(f"Merging {len(roots)} EPG source(s)")
    merged = merge_xmltv_elements(roots)

    # Count results
    stats["channels"] = len(merged.findall('.//channel'))
    stats["programmes"] = len(merged.findall('.//programme'))

    # Write output
    tree = ET.ElementTree(merged)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_str = str(output_path)

    if out_str.endswith('.gz'):
        with gzip.open(out_str, 'wb') as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
    else:
        tree.write(out_str, encoding="utf-8", xml_declaration=True)

    logging.info(f"Written merged EPG to {output_path}: {stats['channels']} channels, {stats['programmes']} programmes")

    return {
        "status": "ok",
        "channels": stats["channels"],
        "events": stats["programmes"],
        "output": str(output_path),
        "stats": stats
    }


# ---------------- DB Lookup ----------------
ALLOWED_LOOKUP_COLUMNS = {"country_name", "country", "station_name", "lineup_name", "lineupid"}

def get_lineups_from_column(column: str, value: str, max_lineups: int = None,
                           db_path: str = DB_FILE, ota_only: bool = False) -> List[str]:
    """Query database for lineups matching a column value."""
    if column not in ALLOWED_LOOKUP_COLUMNS:
        logging.error(f"Invalid lookup column: {column}. Allowed: {ALLOWED_LOOKUP_COLUMNS}")
        return []

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
                    "youtube", "gnstr", "imdbtv","samsung", "pluto", "tubitv", "rakuten", "slingtv"]
    result = []
    for lid in lineup_ids:
        lid_lower = lid.lower()
        if any(bad in lid_lower for bad in bad_patterns):
            logging.warning(f"Filtered out broken lineup: {lid}")
            continue
        result.append(lid)
    return result


# ---------------- Database Update Functions ----------------
def init_db_schema(db_path: str):
    """Initialize or update database schema for lineup discovery."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure channels_by_country table exists (main table used by epg2.py)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS channels_by_country (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lineupid TEXT NOT NULL,
            lineup_name TEXT,
            stationid TEXT,
            callsign TEXT,
            station_name TEXT,
            affiliatecallsign TEXT,
            affiliateid TEXT,
            country TEXT NOT NULL,
            country_name TEXT,
            channel_number TEXT,
            channel_position INTEGER,
            last_updated TIMESTAMP
        )
    ''')

    # Create lineups table for tracking discovered lineups
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lineups (
            lineup_id TEXT PRIMARY KEY,
            lineup_name TEXT,
            lineup_type TEXT,
            lineup_location TEXT,
            country_code TEXT NOT NULL,
            postal_code TEXT,
            device TEXT,
            channel_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP,
            last_fetched TIMESTAMP
        )
    ''')

    # Create postal_codes table for tracking what we've queried
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS postal_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country_code TEXT NOT NULL,
            postal_code TEXT NOT NULL,
            lineup_count INTEGER DEFAULT 0,
            last_queried TIMESTAMP,
            UNIQUE(country_code, postal_code)
        )
    ''')

    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_channels_lineup ON channels_by_country(lineupid)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_channels_country ON channels_by_country(country)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_channels_station ON channels_by_country(stationid)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_lineups_country ON lineups(country_code)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_postal_country ON postal_codes(country_code)')

    conn.commit()
    conn.close()
    logging.info(f"Database schema initialized: {db_path}")


def discover_lineups_for_postal(country: str, postal_code: str) -> List[Dict]:
    """
    Discover available lineups for a postal code.

    Uses two methods:
    1. Try the Gracenote provider API (may be blocked by WAF)
    2. Generate OTA lineup patterns and validate via grid API

    Returns list of lineup dictionaries with lineup_id, name, type, location.
    """
    lineups = []
    seen_ids = set()

    # Method 1: Try provider API (often blocked by AWS WAF now)
    url = f"{PROVIDER_URL}/{postal_code}/{country}"
    headers = {
        'User-Agent': secrets.choice(USER_AGENTS),
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://tvlistings.gracenote.com/',
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list):
                    for provider in data:
                        lineup_id = provider.get('lineupId', '')
                        if lineup_id and lineup_id not in seen_ids:
                            lineups.append({
                                'lineup_id': lineup_id,
                                'name': provider.get('name', 'Unknown'),
                                'type': provider.get('type', 'Unknown'),
                                'location': provider.get('location', postal_code),
                            })
                            seen_ids.add(lineup_id)
                            logging.debug(f"  API found: {lineup_id}")
            except json.JSONDecodeError:
                pass
    except Exception:
        pass  # Provider API often fails, continue to method 2

    # Method 2: Validate OTA lineup pattern via grid API
    # OTA lineups use format: {COUNTRY}-OTA-{POSTAL}
    ota_lineup_id = f"{country}-OTA-{postal_code}"
    if ota_lineup_id not in seen_ids:
        if _validate_lineup_via_grid(country, ota_lineup_id, postal_code):
            lineups.append({
                'lineup_id': ota_lineup_id,
                'name': 'Antenna (OTA)',
                'type': 'Antenna',
                'location': postal_code,
            })
            seen_ids.add(ota_lineup_id)
            logging.debug(f"  Validated OTA lineup: {ota_lineup_id}")

    return lineups


def _validate_lineup_via_grid(country: str, lineup_id: str, postal_code: str) -> bool:
    """
    Validate a lineup ID by attempting to fetch grid data.
    Returns True if the lineup returns valid channel data.
    """
    try:
        _channels, stats = fetch_grid(country, lineup_id, postal_code, timespan=1)
        return stats.get('channels_found', 0) > 0
    except Exception as e:
        logging.debug(f"Validation failed for {lineup_id}: {e}")
        return False


def fetch_channels_for_lineup(lineup_id: str, country: str, postal_code: str = "00000") -> List[Dict]:
    """
    Fetch channel list for a specific lineup using the grid API.
    Returns list of channel dictionaries.
    """
    channels = []

    try:
        # fetch_grid returns (channels_list, stats_dict)
        channel_list, stats = fetch_grid(country, lineup_id, postal_code, timespan=1)

        if channel_list and stats.get('channels_found', 0) > 0:
            for ch in channel_list:
                channel = {
                    'stationid': ch.get('stationId', ''),
                    'callsign': ch.get('callSign', ''),
                    'station_name': ch.get('affiliateName', '') or ch.get('callSign', ''),
                    'affiliatecallsign': ch.get('affiliateName', ''),
                    'affiliateid': '',
                    'channel_number': ch.get('channelNo', ''),
                }
                channels.append(channel)

    except Exception as e:
        logging.debug(f"Could not fetch channels for {lineup_id}: {e}")

    return channels


def save_lineup_to_db(db_path: str, lineup: Dict, country: str, postal_code: str):
    """Save a discovered lineup to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO lineups
        (lineup_id, lineup_name, lineup_type, lineup_location, country_code, postal_code, device, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        lineup['lineup_id'],
        lineup.get('name', ''),
        lineup.get('type', ''),
        lineup.get('location', ''),
        country,
        postal_code,
        lineup.get('device', ''),
        _dt.datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def save_channels_to_db(db_path: str, lineup_id: str, lineup_name: str,
                        channels: List[Dict], country: str, country_name: str):
    """Save channel list for a lineup to the database (matches existing schema)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # First, remove existing channels for this lineup to avoid duplicates
    cursor.execute('DELETE FROM channels_by_country WHERE lineupid = ?', (lineup_id,))

    # Insert new channels (using existing table schema without extra columns)
    for ch in channels:
        cursor.execute('''
            INSERT INTO channels_by_country
            (lineupid, lineup_name, stationid, callsign, station_name,
             affiliatecallsign, affiliateid, country, country_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            lineup_id,
            lineup_name,
            ch.get('stationid', ''),
            ch.get('callsign', ''),
            ch.get('station_name', ''),
            ch.get('affiliatecallsign', ''),
            ch.get('affiliateid', ''),
            country,
            country_name,
        ))

    # Update lineup metadata table if it exists
    try:
        cursor.execute('''
            UPDATE lineups SET channel_count = ?, last_fetched = ? WHERE lineup_id = ?
        ''', (len(channels), _dt.datetime.now().isoformat(), lineup_id))
    except sqlite3.OperationalError:
        pass  # lineups table may not exist in older databases

    conn.commit()
    conn.close()


def save_postal_query(db_path: str, country: str, postal_code: str, lineup_count: int):
    """Record that we queried a postal code."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO postal_codes (country_code, postal_code, lineup_count, last_queried)
        VALUES (?, ?, ?, ?)
    ''', (country, postal_code, lineup_count, _dt.datetime.now().isoformat()))

    conn.commit()
    conn.close()


def load_postal_codes_from_file(filepath: str) -> Dict[str, List[str]]:
    """Load postal codes from a CSV file. Expected columns: country_code, postal_code"""
    import csv
    postal_codes = {}

    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                country = row.get('country_code', row.get('country', '')).upper().strip()
                postal = row.get('postal_code', row.get('zip', '')).strip()

                if country and postal:
                    if country not in postal_codes:
                        postal_codes[country] = []
                    if postal not in postal_codes[country]:
                        postal_codes[country].append(postal)

        logging.info(f"Loaded postal codes for {len(postal_codes)} countries from {filepath}")

    except FileNotFoundError:
        logging.error(f"Postal codes file not found: {filepath}")
    except Exception as e:
        logging.error(f"Error loading postal codes file: {e}")

    return postal_codes


def remove_lineup_from_db(db_path: str, lineup_id: str):
    """Remove a broken lineup and its channels from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Remove from channels_by_country
    cursor.execute('DELETE FROM channels_by_country WHERE lineupid = ?', (lineup_id,))
    channels_removed = cursor.rowcount

    # Remove from lineups table if it exists
    cursor.execute('DELETE FROM lineups WHERE lineup_id = ?', (lineup_id,))

    conn.commit()
    conn.close()

    if channels_removed > 0:
        logging.info(f"Removed broken lineup {lineup_id} ({channels_removed} channel entries)")

    return channels_removed


def run_database_update(db_path: str, countries: List[str] = None,
                        postal_codes_file: str = None, rate_limit: float = 1.0,
                        validate_existing: bool = True, max_workers: int = 1):
    """
    Main function to update the lineup database by scraping Gracenote.

    - Discovers new lineups and fetches their channels
    - Adds channels directly to channels_by_country (used by main script)
    - Removes broken lineups that no longer work

    Args:
        db_path: Path to SQLite database
        countries: List of country codes to process (default: all GRACENOTE_COUNTRIES)
        postal_codes_file: Optional CSV file with postal codes
        rate_limit: Seconds between API requests
        validate_existing: Also validate existing lineups and remove broken ones
        max_workers: Number of parallel workers (from EPG_MAX_WORKERS)
    """
    import threading

    print("\n" + "="*70)
    print("GRACENOTE LINEUP DATABASE UPDATE")
    print("="*70)

    # Initialize database schema
    init_db_schema(db_path)

    # Determine which countries to process
    if countries:
        country_list = [c.upper() for c in countries if c.upper() in GRACENOTE_COUNTRIES]
        if not country_list:
            logging.error(f"No valid countries specified. Available: {', '.join(GRACENOTE_COUNTRIES.keys())}")
            return False
    else:
        country_list = list(GRACENOTE_COUNTRIES.keys())

    print(f"Processing {len(country_list)} countries")
    if max_workers > 1:
        print(f"Using {max_workers} parallel workers")

    # Load postal codes
    if postal_codes_file:
        postal_codes_by_country = load_postal_codes_from_file(postal_codes_file)
    else:
        # Use built-in sample postal codes
        postal_codes_by_country = {
            country: info['sample_postals']
            for country, info in GRACENOTE_COUNTRIES.items()
        }
        print("Using built-in sample postal codes")
        print("(Provide --postal-codes-file for more comprehensive coverage)")

    # Calculate total postal codes for progress tracking
    total_postals = sum(
        len(postal_codes_by_country.get(c, GRACENOTE_COUNTRIES[c].get('sample_postals', [])))
        for c in country_list
    )
    completed_postals = 0

    print(f"Total postal codes to process: {total_postals}")

    # Thread-safe statistics
    stats = {
        'countries': 0,
        'postal_codes': 0,
        'lineups_added': 0,
        'lineups_updated': 0,
        'lineups_removed': 0,
        'channels_added': 0,
        'errors': 0,
    }
    stats_lock = threading.Lock()
    print_lock = threading.Lock()

    start_time = time.time()

    def format_eta(elapsed: float, completed: int, total: int) -> str:
        """Calculate and format ETA."""
        if completed == 0:
            return "calculating..."
        rate = elapsed / completed
        remaining = (total - completed) * rate
        if remaining < 60:
            return f"{int(remaining)}s"
        elif remaining < 3600:
            return f"{int(remaining // 60)}m {int(remaining % 60)}s"
        else:
            return f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"

    # Get existing lineups from channels_by_country (the table the script actually uses)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT lineupid FROM channels_by_country')
    existing_lineups = set(row[0] for row in cursor.fetchall())
    conn.close()
    existing_lineups_lock = threading.Lock()

    print(f"Existing lineups in database: {len(existing_lineups)}")

    # Track lineups we've validated as working
    validated_lineups = set()
    validated_lock = threading.Lock()

    def process_postal(country: str, country_name: str, postal: str) -> dict:
        """Process a single postal code (thread-safe)."""
        result = {'added': 0, 'updated': 0, 'channels': 0, 'error': False}

        try:
            lineups = discover_lineups_for_postal(country, postal)

            if lineups:
                for lineup in lineups:
                    lineup_id = lineup['lineup_id']

                    # Skip known broken patterns
                    if any(bad in lineup_id.lower() for bad in
                           ["xumotv", "sandbox", "test", "dummy", "dtvnow", "amzpv"]):
                        continue

                    with validated_lock:
                        validated_lineups.add(lineup_id)

                    # Fetch channels for this lineup
                    time.sleep(rate_limit * 0.5)
                    channels = fetch_channels_for_lineup(lineup_id, country, postal)

                    if channels:
                        with existing_lineups_lock:
                            is_new = lineup_id not in existing_lineups
                            existing_lineups.add(lineup_id)

                        # Save to database
                        save_channels_to_db(db_path, lineup_id, lineup['name'],
                                           channels, country, country_name)
                        save_lineup_to_db(db_path, lineup, country, postal)

                        result['channels'] += len(channels)
                        if is_new:
                            result['added'] += 1
                            with print_lock:
                                print(f"    + {lineup_id}: {len(channels)} channels (NEW)")
                        else:
                            result['updated'] += 1
                            with print_lock:
                                print(f"    ~ {lineup_id}: {len(channels)} channels (updated)")

            save_postal_query(db_path, country, postal, len(lineups))

        except Exception as e:
            result['error'] = True
            with print_lock:
                print(f"    error: {e}")

        return result

    # Process each country
    for country in country_list:
        country_info = GRACENOTE_COUNTRIES[country]
        country_name = country_info['name']

        print(f"\n{'─'*50}")
        print(f"[{country}] {country_name}")
        print(f"{'─'*50}")

        postal_codes = postal_codes_by_country.get(country, [])

        if not postal_codes:
            postal_codes = country_info.get('sample_postals', [])

        if not postal_codes:
            print(f"  No postal codes available, skipping")
            continue

        stats['countries'] += 1
        country_added = 0
        country_updated = 0

        if max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Print all postal codes being queried
                for postal in postal_codes:
                    print(f"  Querying {postal}...")

                futures = {
                    executor.submit(process_postal, country, country_name, postal): postal
                    for postal in postal_codes
                }

                for future in as_completed(futures):
                    postal = futures[future]
                    try:
                        result = future.result()
                        stats['postal_codes'] += 1
                        stats['channels_added'] += result['channels']
                        stats['lineups_added'] += result['added']
                        stats['lineups_updated'] += result['updated']
                        country_added += result['added']
                        country_updated += result['updated']
                        if result['error']:
                            stats['errors'] += 1
                    except Exception as e:
                        with print_lock:
                            print(f"  {postal} error: {e}")
                        stats['errors'] += 1
        else:
            # Sequential processing
            for postal in postal_codes:
                print(f"  Querying {postal}...", end=" ", flush=True)

                try:
                    lineups = discover_lineups_for_postal(country, postal)
                    stats['postal_codes'] += 1

                    if lineups:
                        print(f"found {len(lineups)} lineup(s)")

                        for lineup in lineups:
                            lineup_id = lineup['lineup_id']

                            if any(bad in lineup_id.lower() for bad in
                                   ["xumotv", "sandbox", "test", "dummy", "dtvnow", "amzpv"]):
                                continue

                            validated_lineups.add(lineup_id)

                            time.sleep(rate_limit * 0.5)
                            channels = fetch_channels_for_lineup(lineup_id, country, postal)

                            if channels:
                                is_new = lineup_id not in existing_lineups
                                save_channels_to_db(db_path, lineup_id, lineup['name'],
                                                   channels, country, country_name)
                                stats['channels_added'] += len(channels)

                                if is_new:
                                    stats['lineups_added'] += 1
                                    country_added += 1
                                    existing_lineups.add(lineup_id)
                                    print(f"    + {lineup_id}: {len(channels)} channels (NEW)")
                                else:
                                    stats['lineups_updated'] += 1
                                    country_updated += 1
                                    print(f"    ~ {lineup_id}: {len(channels)} channels (updated)")

                                save_lineup_to_db(db_path, lineup, country, postal)
                    else:
                        print("no lineups")

                    save_postal_query(db_path, country, postal, len(lineups))

                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"error: {e}")
                    stats['errors'] += 1

                time.sleep(rate_limit)

        completed_postals += len(postal_codes)
        elapsed = time.time() - start_time
        eta = format_eta(elapsed, completed_postals, total_postals)
        pct = int(100 * completed_postals / total_postals) if total_postals > 0 else 100
        print(f"  Country: +{country_added} new, ~{country_updated} updated")
        print(f"  Progress: [{completed_postals}/{total_postals}] {pct}% - ETA: {eta}")

    # Validate and remove broken lineups from existing database
    if validate_existing and existing_lineups:
        print(f"\n{'─'*50}")
        print("Validating existing lineups...")
        print(f"{'─'*50}")

        # Check lineups that we didn't already validate during discovery
        lineups_to_check = existing_lineups - validated_lineups

        # Build list of lineups to validate with their country info
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        validation_queue = []

        for lineup_id in list(lineups_to_check):
            cursor.execute('SELECT country FROM channels_by_country WHERE lineupid = ? LIMIT 1', (lineup_id,))
            row = cursor.fetchone()
            if row:
                lineup_country = row[0]
                if lineup_country in country_list:
                    parts = lineup_id.split('-')
                    postal = parts[-1] if len(parts) >= 3 else '00000'
                    validation_queue.append((lineup_id, lineup_country, postal))

        conn.close()

        total_to_validate = len(validation_queue)
        validated_count = 0
        validation_start = time.time()

        # Limit validation workers to avoid 429 rate limits
        validation_workers = min(max_workers, 3)
        print(f"  {total_to_validate} lineups to validate ({validation_workers} workers)")

        def validate_one(item):
            """Validate a single lineup with built-in delay."""
            lineup_id, lineup_country, postal = item
            time.sleep(rate_limit * 0.5)  # Delay to avoid 429
            is_valid = _validate_lineup_via_grid(lineup_country, lineup_id, postal)
            return lineup_id, is_valid

        if validation_workers > 1 and total_to_validate > 0:
            # Parallel validation with rate limiting
            removed_count = 0
            with ThreadPoolExecutor(max_workers=validation_workers) as executor:
                futures = {executor.submit(validate_one, item): item for item in validation_queue}

                for future in as_completed(futures):
                    lineup_id, is_valid = future.result()
                    validated_count += 1

                    if is_valid:
                        with validated_lock:
                            validated_lineups.add(lineup_id)
                    else:
                        removed = remove_lineup_from_db(db_path, lineup_id)
                        if removed > 0:
                            with stats_lock:
                                stats['lineups_removed'] += 1
                            removed_count += 1
                            with print_lock:
                                print(f"    Removed: {lineup_id}")

                    # Progress update every 50 or at end
                    if validated_count % 50 == 0 or validated_count == total_to_validate:
                        elapsed_v = time.time() - validation_start
                        eta_v = format_eta(elapsed_v, validated_count, total_to_validate)
                        pct_v = int(100 * validated_count / total_to_validate)
                        with print_lock:
                            print(f"  Validation: [{validated_count}/{total_to_validate}] {pct_v}% - ETA: {eta_v} - Removed: {removed_count}")
        else:
            # Sequential validation
            for lineup_id, lineup_country, postal in validation_queue:
                print(f"  Checking {lineup_id}...", end=" ", flush=True)

                if _validate_lineup_via_grid(lineup_country, lineup_id, postal):
                    print("OK")
                    validated_lineups.add(lineup_id)
                else:
                    print("BROKEN - removing")
                    removed = remove_lineup_from_db(db_path, lineup_id)
                    if removed > 0:
                        stats['lineups_removed'] += 1

                validated_count += 1
                if validated_count % 20 == 0:
                    elapsed_v = time.time() - validation_start
                    eta_v = format_eta(elapsed_v, validated_count, total_to_validate)
                    pct_v = int(100 * validated_count / total_to_validate)
                    print(f"  Progress: [{validated_count}/{total_to_validate}] {pct_v}% - ETA: {eta_v}")

                time.sleep(rate_limit)

    # Print final statistics
    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("UPDATE COMPLETE")
    print("="*70)
    print(f"Time elapsed:       {elapsed:.1f} seconds")
    print(f"Countries:          {stats['countries']}")
    print(f"Postal codes:       {stats['postal_codes']}")
    print(f"Lineups added:      {stats['lineups_added']}")
    print(f"Lineups updated:    {stats['lineups_updated']}")
    print(f"Lineups removed:    {stats['lineups_removed']}")
    print(f"Channels added:     {stats['channels_added']}")
    print(f"Errors:             {stats['errors']}")

    # Show database totals
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(DISTINCT lineupid) FROM channels_by_country')
    total_lineups = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM channels_by_country')
    total_channels = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT country) FROM channels_by_country')
    total_countries = cursor.fetchone()[0]
    conn.close()

    print(f"\nDatabase totals:")
    print(f"  Lineups:          {total_lineups}")
    print(f"  Channel entries:  {total_channels}")
    print(f"  Countries:        {total_countries}")
    print("="*70 + "\n")

    return True


# ---------------- Profile & Config Helpers ----------------
def resolve_lineups_for_profile(profile: Dict, base_config: Dict) -> List[str]:
    """Resolve lineup IDs for a single profile."""
    mode = profile.get("lookup_mode", "").lower()
    raw_val = profile.get("lookup_value", "")
    ota_only = profile.get("ota_only", False)
    max_lineups = profile.get("max_lineups")
    db_path = base_config.get("EPG_DB_PATH", DB_FILE)

    if not mode or not raw_val:
        return []

    lookup_values = [v.strip() for v in raw_val.split(",") if v.strip()]
    all_lineups = []

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

    filtered = filter_broken_lineups(all_lineups)
    unique = list(dict.fromkeys(filtered))

    if max_lineups and len(unique) > max_lineups:
        unique = unique[:max_lineups]

    return unique


def load_profiles(config_path: str = CONFIG_FILE) -> List[Dict]:
    """Load profiles from config file."""
    if not Path(config_path).exists():
        return []

    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return cfg.get("EPG_PROFILES", [])
    except Exception:
        return []


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
        "EPG_FALLBACK_ZIP": "",
    }
    
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                file_cfg = json.load(f)
            # Only update config keys that don't start with underscore (skip _comment, _notes, etc.)
            for k, v in file_cfg.items():
                if not k.startswith("_"):
                    config[k.upper()] = v
        except Exception as e:
            # Print to stderr since logging may not be set up yet
            print(f"Warning: Failed to read {CONFIG_FILE}: {e}", file=sys.stderr)
    
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
        if args.days is not None:
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
        if args.fallback_zip:
            config["EPG_FALLBACK_ZIP"] = args.fallback_zip

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
def _derive_country_from_lineup(lineup_id: str, default_country: str = "USA") -> str:
    """Derive country code from lineup ID prefix (e.g., CAN-OTA... -> CAN, DEU-... -> DEU)."""
    if not lineup_id:
        return default_country
    upper_id = lineup_id.upper()
    # Check for known country prefixes (3-letter ISO codes)
    # North America
    KNOWN_PREFIXES = [
        "USA", "CAN", "MEX",
        # Europe
        "DEU", "FRA", "ESP", "ITA", "GBR", "NLD", "BEL", "AUT", "CHE",
        "POL", "SWE", "NOR", "DNK", "FIN", "IRL", "PRT",
        # Latin America
        "ARG", "BRA", "CHL", "COL", "PER", "ECU", "VEN", "CRI", "PAN",
        "GTM", "HND", "SLV", "NIC", "BOL", "PRY", "URY", "DOM",
        # Caribbean
        "JAM", "BHS", "TTO", "CYM", "CUW", "ABW", "BLZ", "PRI",
        # Other
        "AUS", "NZL"
    ]
    for prefix in KNOWN_PREFIXES:
        if upper_id.startswith(prefix + "-"):
            return prefix
    return default_country


def fetch_lineup_wrapper(args: Tuple) -> Tuple[str, List[Dict], Dict]:
    """Wrapper for parallel execution."""
    lineup, country, postal, timespan, delay, retry_count, fallback_zip = args

    # Derive country from lineup ID for proper API handling
    derived_country = _derive_country_from_lineup(lineup, country)

    if not postal:
        zip_lookup = lookup_zip_for_lineup(lineup)
        if zip_lookup:
            postal = zip_lookup
        elif fallback_zip:
            postal = fallback_zip
            logging.debug(f"Using fallback ZIP {fallback_zip} for {lineup}")

    try:
        channels, stats = fetch_grid(
            country=derived_country,
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
def lookup_zip_for_lineup(lineup_id: str, db_path: str = DB_FILE) -> Optional[str]:
    """Get representative ZIP for each lineup from database or extract from lineup ID."""
    if lineup_id:
        upper_id = lineup_id.upper()

        # USA patterns: USA-OTA12345, USA-CA90210-X, USA-LOCALBROADCAST-63601
        if upper_id.startswith("USA"):
            # Try USA-OTA<ZIP> pattern first (e.g., USA-OTA12345)
            match = re.search(r'-OTA(\d{5})', upper_id)
            if match:
                return match.group(1)
            # Try USA-LOCALBROADCAST-<ZIP> pattern
            match = re.search(r'-LOCALBROADCAST-(\d{5})', upper_id)
            if match:
                return match.group(1)
            # Fallback: extract any 5-digit ZIP from the ID
            match = re.search(r'(\d{5})', lineup_id)
            if match:
                return match.group(1)

        # Canadian patterns: CAN-OTAJ9X0A1 -> J9X 0A1, CAN-OTAK1P5W1 -> K1P 5W1
        elif upper_id.startswith("CAN"):
            # Extract everything after -OTA (Canadian postal codes are 6 alphanumeric chars)
            match = re.search(r'-OTA([A-Za-z0-9]+)', lineup_id, re.I)
            if match:
                postal = match.group(1)
                # Canadian postal codes are 6 chars: A1A1A1 -> A1A 1A1
                if len(postal) >= 6:
                    return f"{postal[:3]} {postal[3:6]}"
                elif len(postal) >= 3:
                    return postal[:3]  # Return partial if available

        # UK patterns: GBR-1000031-DEFAULT (numeric, no postal code embedded)
        elif upper_id.startswith("GBR"):
            # UK postal codes are not embedded in lineup IDs
            # Return a central London default
            return "SW1A 1AA"

    # Try database lookup (may not have postal_code column)
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # Check if postal_code column exists
        cur.execute("PRAGMA table_info(channels_by_country)")
        columns = [row[1] for row in cur.fetchall()]

        if 'postal_code' in columns:
            cur.execute("SELECT postal_code FROM channels_by_country WHERE lineupid=? LIMIT 1", (lineup_id,))
            row = cur.fetchone()
            if row and row[0]:
                conn.close()
                return row[0]
        conn.close()
    except Exception as e:
        logging.debug(f"ZIP lookup failed for {lineup_id}: {e}")

    # Country-specific default ZIPs for OTA
    if lineup_id:
        if lineup_id.startswith("USA"):
            return "10001"  # NYC
        elif lineup_id.startswith("CAN"):
            return "M5V 3K9"  # Toronto
        elif lineup_id.startswith("GBR"):
            return "SW1A 1AA"  # London

    return None

def print_summary_table(all_stats: Dict, total_channels: int, total_events: int, output_file: Path):
    """Print a formatted summary table of the fetch results."""
    print("\n" + "="*70)
    print("EPG FETCH SUMMARY")
    print("="*70)

    # Column headers
    print(f"{'Lineup ID':<40} {'Status':<10} {'Channels':>8} {'Events':>10}")
    print("-"*70)

    for lineup_id, stats in sorted(all_stats.items()):
        if "error" in stats:
            status = "FAILED"
            ch_count = "-"
            ev_count = "-"
        else:
            status = "OK"
            ch_count = str(stats.get("channels_found", 0))
            ev_count = str(stats.get("events_found", 0))

        # Truncate long lineup IDs
        display_id = lineup_id[:38] + ".." if len(lineup_id) > 40 else lineup_id
        print(f"{display_id:<40} {status:<10} {ch_count:>8} {ev_count:>10}")

    print("-"*70)
    successful = sum(1 for s in all_stats.values() if "error" not in s)
    failed = len(all_stats) - successful
    print(f"{'TOTAL':<40} {f'{successful}/{len(all_stats)}':<10} {total_channels:>8} {total_events:>10}")
    print("="*70)
    print(f"Output: {output_file}")
    if failed > 0:
        print(f"Warning: {failed} lineup(s) failed to fetch")
    print("")

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
    fallback_zip = config.get("EPG_FALLBACK_ZIP", "")

    logging.info(f"Starting EPG fetch for {len(lineup_ids)} lineup(s)")
    logging.info(f"  Timespan: {timespan} hours ({timespan // 24} days)")
    logging.info(f"  Parallel: {parallel}" + (f" (workers: {max_workers})" if parallel else ""))

    all_channels = []
    all_stats = {}
    country_suffix_map = {}
    completed_count = 0

    if parallel and len(lineup_ids) > 1:
        args_list = [
            (lid, country, postal, timespan, delay, retry_count, fallback_zip)
            for lid in lineup_ids
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_lineup_wrapper, args): args[0] for args in args_list}

            for future in as_completed(futures):
                lineup = futures[future]
                completed_count += 1
                progress = f"[{completed_count}/{len(lineup_ids)}]"
                try:
                    lineup_id, channels, stats = future.result()
                    all_channels.extend(channels)
                    all_stats[lineup_id] = stats

                    suffix = get_country_suffix(lineup_id, db_path)
                    for ch in channels:
                        station_id = str(ch.get("stationId") or ch.get("channelId") or "")
                        if station_id:
                            country_suffix_map[station_id] = suffix

                    logging.info(f"{progress} {lineup_id}: {stats.get('channels_found', 0)} channels, "
                               f"{stats.get('events_found', 0)} events")
                except Exception as e:
                    logging.error(f"{progress} {lineup}: FAILED - {e}")
                    all_stats[lineup] = {"error": str(e)}
    else:
        for idx, lineup in enumerate(lineup_ids, 1):
            progress = f"[{idx}/{len(lineup_ids)}]"
            logging.info(f"{progress} Fetching: {lineup}")

            # Derive country from lineup ID for proper API handling
            derived_country = _derive_country_from_lineup(lineup, country)

            # Get postal code if not provided
            lineup_postal = postal
            if not lineup_postal:
                lineup_postal = lookup_zip_for_lineup(lineup)
            if not lineup_postal and fallback_zip:
                lineup_postal = fallback_zip

            try:
                channels, stats = fetch_grid(
                    country=derived_country,
                    lineup_id_input=lineup,
                    postal=lineup_postal,
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

                logging.info(f"{progress} {lineup}: {stats.get('channels_found', 0)} channels, "
                           f"{stats.get('events_found', 0)} events")
            except Exception as e:
                logging.error(f"{progress} {lineup}: FAILED - {e}")
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

    # Print summary table
    print_summary_table(all_stats, total_channels, total_events, output_file)

    return {"channels": total_channels, "events": total_events, "output": str(output_file)}


def run_all_profiles(base_config: Dict):
    """Run EPG fetch for all configured profiles."""
    profiles = load_profiles()

    if not profiles:
        logging.warning("No profiles found in config. Running single fetch mode.")
        run_epg(base_config)
        return

    print("\n" + "="*70)
    print(f"MULTI-PROFILE MODE - {len(profiles)} profile(s) configured")
    print("="*70)

    results = []
    total_start = time.time()

    for idx, profile in enumerate(profiles, 1):
        name = profile.get("name", f"Profile {idx}")
        output = profile.get("output", f"EPG-{idx}.xml")
        source_type = profile.get("source", "gracenote").lower()

        print(f"\n{'─'*70}")
        print(f"PROFILE {idx}/{len(profiles)}: {name}")
        print(f"{'─'*70}")

        # Handle external sources (globetvapp, mjh, direct URLs)
        if source_type in ("external", "globetvapp", "mjh") or profile.get("urls"):
            logging.info(f"[{name}] External source mode: {source_type}")
            try:
                result = fetch_and_merge_external(
                    profile,
                    Path(output),
                    max_retries=int(base_config.get("EPG_RETRY_COUNT", 3))
                )
                if result.get("status") == "ok":
                    results.append({
                        "name": name,
                        "status": "OK",
                        "channels": result.get("channels", 0),
                        "events": result.get("events", 0),
                        "output": result.get("output", output)
                    })
                else:
                    results.append({
                        "name": name,
                        "status": "FAILED",
                        "reason": result.get("message", "Unknown error")
                    })
            except Exception as e:
                logging.error(f"[{name}] External fetch error: {e}")
                results.append({"name": name, "status": "FAILED", "reason": str(e)})
            continue

        # Handle GraceNote sources (original behavior)
        # Resolve lineups for this profile
        lineups = resolve_lineups_for_profile(profile, base_config)

        if not lineups:
            logging.warning(f"[{name}] No lineups found, skipping.")
            results.append({"name": name, "status": "SKIPPED", "reason": "No lineups found"})
            continue

        logging.info(f"[{name}] Found {len(lineups)} lineup(s)")

        # Build profile-specific config
        profile_config = base_config.copy()
        profile_config["EPG_LINEUP_ID"] = ",".join(lineups)
        profile_config["EPG_OUTPUT"] = output
        profile_config["EPG_OTA_ONLY"] = profile.get("ota_only", False)

        try:
            result = run_epg(profile_config)
            results.append({
                "name": name,
                "status": "OK",
                "channels": result["channels"],
                "events": result["events"],
                "output": result["output"]
            })
        except SystemExit:
            results.append({"name": name, "status": "FAILED", "reason": "Fetch error"})
        except Exception as e:
            logging.error(f"[{name}] Error: {e}")
            results.append({"name": name, "status": "FAILED", "reason": str(e)})

    # Print final multi-profile summary
    elapsed = time.time() - total_start
    print("\n" + "="*70)
    print("MULTI-PROFILE SUMMARY")
    print("="*70)
    print(f"{'Profile':<25} {'Status':<10} {'Channels':>10} {'Events':>12} {'Output':<20}")
    print("-"*70)

    for r in results:
        name = r["name"][:23] + ".." if len(r["name"]) > 25 else r["name"]
        status = r["status"]
        channels = str(r.get("channels", "-"))
        events = str(r.get("events", "-"))
        output = r.get("output", r.get("reason", "-"))
        if len(output) > 18:
            output = ".." + output[-16:]
        print(f"{name:<25} {status:<10} {channels:>10} {events:>12} {output:<20}")

    print("-"*70)
    succeeded = sum(1 for r in results if r["status"] == "OK")
    print(f"Completed: {succeeded}/{len(results)} profiles in {elapsed:.1f}s")
    print("="*70 + "\n")


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
    location_group.add_argument('--country', type=str, default='USA',
                               help='Country code (default: %(default)s)')
    location_group.add_argument('--zip', type=str,
                               help='Postal/ZIP code (auto-detected if not set)')
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output', '-o', type=str, default='EPG.xml',
                             help='Output XMLTV file path (default: %(default)s)')
    output_group.add_argument('--no-thumbnails', action='store_true',
                             help='Exclude thumbnail/icon URLs from output')

    fetch_group = parser.add_argument_group('Fetching Options')
    fetch_group.add_argument('--days', type=int, default=None,
                            help='Number of days to fetch, 1-7 (default: from config or 1)')
    fetch_group.add_argument('--delay', type=int, default=0,
                            help='Delay in seconds between API requests (default: %(default)s)')
    fetch_group.add_argument('--retry-count', type=int, default=3,
                            help='Number of retries for failed requests (default: %(default)s)')
    fetch_group.add_argument('--max-lineups', type=int,
                            help='Maximum number of lineups to fetch from database lookup')

    parallel_group = parser.add_argument_group('Parallel Execution')
    parallel_group.add_argument('--parallel', action='store_true',
                               help='Fetch multiple lineups in parallel')
    parallel_group.add_argument('--max-workers', type=int, default=3,
                               help='Maximum parallel workers (default: %(default)s)')

    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--profiles', action='store_true',
                           help='Run all profiles defined in config (multi-output mode)')
    misc_group.add_argument('--dry-run', action='store_true',
                           help='Show what would be fetched without making API calls')
    misc_group.add_argument('--init-config', action='store_true',
                           help='Generate a sample config file and exit')
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--log-level', type=str,
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Logging level')
    log_group.add_argument('--log-file', type=str,
                          help='Log file path (logs to console if not specified)')
    
    db_group = parser.add_argument_group('Database')
    db_group.add_argument('--db-path', type=str,
                         help=f'Path to SQLite database (default: {DB_FILE})')
    db_group.add_argument('--update-db', action='store_true',
                         help='Discover new lineups, fetch channels, and remove broken ones')
    db_group.add_argument('--postal-codes-file', type=str,
                         help='CSV file with postal codes (columns: country_code, postal_code)')
    db_group.add_argument('--update-countries', type=str,
                         help='Comma-separated country codes to update (default: all supported)')
    db_group.add_argument('--update-rate-limit', type=float, default=1.0,
                         help='Seconds between API requests during update (default: 1.0)')
    db_group.add_argument('--skip-validation', action='store_true',
                         help='Skip validation of existing lineups (faster, but won\'t remove broken ones)')
    
    cdn_group = parser.add_argument_group('CDN Options')
    cdn_group.add_argument('--cdn-subdomain', type=str,
                          help='TMS Image CDN subdomain (default: zap2it)')

    advanced_group = parser.add_argument_group('Advanced')
    advanced_group.add_argument('--fallback-zip', type=str,
                               help='Fallback ZIP code when auto-lookup fails (default: none)')

    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    return parser.parse_args()

# ---------------- Init Config Helper ----------------
def generate_sample_config():
    """Generate a sample configuration file."""
    sample = {
        "EPG_LINEUP_ID": "",
        "EPG_LOOKUP_MODE": "ota",
        "EPG_LOOKUP_VALUE": "United States",
        "EPG_COUNTRY": "USA",
        "EPG_ZIP": "",
        "EPG_OUTPUT": "EPG.xml",
        "EPG_TIMESPAN_DAYS": 1,
        "EPG_DELAY": 0,
        "EPG_LOG_LEVEL": "INFO",
        "EPG_THUMBNAILS": True,
        "EPG_MAX_LINEUPS": 10,
        "EPG_RETRY_COUNT": 3,
        "EPG_PARALLEL": False,
        "EPG_MAX_WORKERS": 3,
        "EPG_OTA_ONLY": True,
        "EPG_CDN_SUBDOMAIN": "zap2it"
    }

    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        print(f"Config file already exists: {config_path}")
        print("Remove it first if you want to regenerate.")
        return False

    with open(config_path, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Sample config created: {config_path}")
    print("Edit this file to customize your EPG settings.")
    return True

# ---------------- Dry Run ----------------
def dry_run(config: Dict, use_profiles: bool = False):
    """Show what would be fetched without making API calls."""
    print("\n" + "="*60)
    print("DRY RUN - No API calls will be made")
    print("="*60)

    # Debug: show actual loaded values
    timespan_hours = config.get('EPG_TIMESPAN', 24)
    timespan_days = config.get('EPG_TIMESPAN_DAYS', 1)

    print(f"\nGlobal Settings:")
    print(f"  Timespan:     {timespan_hours} hours ({timespan_days} days)")
    print(f"  Thumbnails:   {'Yes' if config.get('EPG_THUMBNAILS', True) else 'No'}")
    print(f"  Parallel:     {'Yes' if config.get('EPG_PARALLEL', False) else 'No'}")
    if config.get('EPG_PARALLEL'):
        print(f"  Max workers:  {config.get('EPG_MAX_WORKERS', 3)}")

    profiles = load_profiles() if use_profiles else []

    if use_profiles and profiles:
        print(f"\n{'─'*60}")
        print(f"MULTI-PROFILE MODE - {len(profiles)} profile(s)")
        print(f"{'─'*60}")

        for idx, profile in enumerate(profiles, 1):
            name = profile.get("name", f"Profile {idx}")
            output = profile.get("output", f"EPG-{idx}.xml")
            mode = profile.get("lookup_mode", "?")
            value = profile.get("lookup_value", "?")
            ota = profile.get("ota_only", False)
            max_l = profile.get("max_lineups", "unlimited")

            lineups = resolve_lineups_for_profile(profile, config)

            print(f"\n  [{idx}] {name}")
            print(f"      Mode: {mode}, Value: {value}")
            print(f"      OTA only: {ota}, Max lineups: {max_l}")
            print(f"      Output: {output}")
            print(f"      Lineups found: {len(lineups)}")
            if lineups and len(lineups) <= 5:
                for lid in lineups:
                    print(f"        - {lid}")
            elif lineups:
                for lid in lineups[:3]:
                    print(f"        - {lid}")
                print(f"        ... and {len(lineups) - 3} more")
    else:
        # Single run mode
        lineup_ids = [lid.strip() for lid in str(config.get("EPG_LINEUP_ID", "")).split(",") if lid.strip()]

        print(f"\nSingle Run Configuration:")
        print(f"  Country:      {config.get('EPG_COUNTRY', 'USA')}")
        print(f"  ZIP/Postal:   {config.get('EPG_ZIP') or '(auto-lookup)'}")
        print(f"  Output file:  {config.get('EPG_OUTPUT', 'EPG.xml')}")

        print(f"\nLineups to fetch ({len(lineup_ids)}):")
        if not lineup_ids:
            print("  (none found - check your lookup settings)")
        else:
            for i, lid in enumerate(lineup_ids, 1):
                ota_marker = " [OTA]" if _is_ota(lid) else ""
                print(f"  {i:3}. {lid}{ota_marker}")

    print("\n" + "="*60)
    cmd = "Run without --dry-run"
    if use_profiles:
        cmd += " (keep --profiles)"
    print(f"{cmd} to execute the fetch.")
    print("="*60 + "\n")

# ---------------- Main ----------------
if __name__ == "__main__":
    args = parse_arguments()

    # Handle --init-config before loading config
    if getattr(args, 'init_config', False):
        success = generate_sample_config()
        sys.exit(0 if success else 1)

    # Handle --update-db (load config first for parallel settings)
    if getattr(args, 'update_db', False):
        config = load_config(args)

        log_level = config.get("EPG_LOG_LEVEL") or getattr(args, 'log_level', 'INFO') or 'INFO'
        setup_logging(log_level)

        db_path = getattr(args, 'db_path', None) or config.get("EPG_DB_PATH") or DB_FILE
        postal_file = getattr(args, 'postal_codes_file', None)
        rate_limit = getattr(args, 'update_rate_limit', 1.0) or 1.0
        skip_validation = getattr(args, 'skip_validation', False)

        # Get parallel settings from config
        use_parallel = config.get("EPG_PARALLEL", False)
        max_workers = config.get("EPG_MAX_WORKERS", 3) if use_parallel else 1

        # Parse countries if specified
        countries = None
        if getattr(args, 'update_countries', None):
            countries = [c.strip().upper() for c in args.update_countries.split(',')]

        success = run_database_update(
            db_path=db_path,
            countries=countries,
            postal_codes_file=postal_file,
            rate_limit=rate_limit,
            validate_existing=not skip_validation,
            max_workers=max_workers
        )
        sys.exit(0 if success else 1)

    config = load_config(args)

    log_level = config.get("EPG_LOG_LEVEL") or config.get("EPG_VERBOSE", 1)
    setup_logging(log_level, config.get("EPG_LOG_FILE"))

    # Handle --dry-run
    if getattr(args, 'dry_run', False):
        dry_run(config, use_profiles=getattr(args, 'profiles', False))
        sys.exit(0)

    try:
        with _single_instance_guard():
            if getattr(args, 'profiles', False):
                run_all_profiles(config)
            else:
                run_epg(config)
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)