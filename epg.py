#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import time
import random
import signal
import fcntl
import atexit
import re
import contextlib
import sqlite3
import datetime as _dt
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------- Global Defaults ----------------
CONFIG_FILE = "/home/sftech13/git/epg/epg_config.json"
DB_FILE = "/home/sftech13/git/epg/zap2it.db"
_LOCK_DIR = "/tmp/epg_standalone.lock.d"
_LOCK_FILE = "/tmp/epg_standalone.run.lock"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
]
BASE_URL = "https://tvlistings.gracenote.com/api/grid"
COUNTRY_3 = {"US": "USA", "CA": "CAN"}

# ---------------- Logging ----------------
def setup_logging(verbosity: int, log_file: str = None):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(fh)

# ---------------- Single Instance Lock ----------------
def _release_locks():
    """Release EPG lock directory and file safely."""
    # Remove lock directory if it exists
    if os.path.isdir(_LOCK_DIR):
        try:
            os.rmdir(_LOCK_DIR)
            logging.debug(f"Released lock directory: {_LOCK_DIR}")
        except OSError as e:
            logging.warning(f"Could not remove lock directory {_LOCK_DIR}: {e}")

    # Remove lock file if it exists
    if os.path.exists(_LOCK_FILE):
        try:
            os.remove(_LOCK_FILE)
            logging.debug(f"Released lock file: {_LOCK_FILE}")
        except OSError as e:
            logging.warning(f"Could not remove lock file {_LOCK_FILE}: {e}")


@contextlib.contextmanager
def _single_instance_guard():
    try:
        os.makedirs('/tmp', exist_ok=True)
        os.mkdir(_LOCK_DIR)
        atexit.register(_release_locks)
        signal.signal(signal.SIGTERM, lambda *a, **k: (_release_locks(), os._exit(0)))
        signal.signal(signal.SIGINT,  lambda *a, **k: (_release_locks(), os._exit(0)))
        yield
        _release_locks()
        return
    except FileExistsError:
        logging.error("Another instance detected. Exiting.")
        raise SystemExit(0)
    try:
        fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode('utf-8'))
        try:
            yield
        finally:
            try:
                os.ftruncate(fd, 0)
                os.close(fd)
            except Exception:
                pass
            try:
                os.remove(_LOCK_FILE)
            except Exception:
                pass
    except BlockingIOError:
        logging.error("Another instance detected. Exiting.")
        raise SystemExit(0)

# ---------------- API helpers ----------------
def _ua():
    return random.choice(USER_AGENTS)

def _is_ota(lineup_id: str) -> bool:
    s = (lineup_id or "").upper()
    return "OTA" in s or "LOCALBROADCAST" in s or s.endswith("-DEFAULT")


def _headend_from_lineup(lineup_id: str) -> str:
    # OTA uses a special headend
    if _is_ota(lineup_id):
        return "lineupId"
    # Cable/Satellite needs only the middle part of the lineup
    m = re.match(r"^[A-Z]{3}-([^-]+)-", lineup_id or "")
    return m.group(1) if m else "lineup"

def _api_lineup_and_headend(country: str, lineup_id: str):
    c3 = COUNTRY_3.get(country.upper(), country.upper())
    if _is_ota(lineup_id):
        return f"{c3}-lineupId-DEFAULT", "lineupId"
    return lineup_id, _headend_from_lineup(lineup_id)


def _device_from_lineup(lineup_id: str) -> str:
    s = (lineup_id or "").upper().strip()
    if _is_ota(s) or s.endswith("-DEFAULT"):
        return "-"
    # The last character (usually letter like X, L, etc.) is device
    m = re.search(r"-([A-Z])$", s)
    return m.group(1) if m else "-"

def _build_url(lineup_id: str, headend_id: str, country: str, postal, time_sec: int, chunk_hours: int, is_ota: bool):
    device = _device_from_lineup(lineup_id)
    user_id = ('%08x' % random.getrandbits(32))

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

    # OTA requires postal code, cable/sat uses "-"
    if is_ota:
        if postal:
            params.insert(6, ("postalCode", str(postal)))
    else:
        params.insert(6, ("postalCode", "-"))

    qs = "&".join(f"{k}={v}" for k, v in params if v not in (None, ""))
    return f"{BASE_URL}?{qs}"

# ---------------- Fetch Logic ----------------
def fetch_grid(country: str, lineup_id_input: str, postal: str, timespan: int = 72, delay_seconds: int = 0, max_retries: int = 3):
    """Fetch EPG data for a given lineup and return channels/events."""
    c3 = COUNTRY_3.get(country.upper(), country.upper())
    lineup_api = lineup_id_input
    headend_api = _headend_from_lineup(lineup_id_input)
    total_hours = int(timespan)
    chunk_hours = 6
    sess = requests.Session()
    channels_map = {}
    base_time = int(time.time())
    offsets = list(range(0, total_hours, chunk_hours))

    for idx, offset in enumerate(offsets):
        t = base_time + offset * 3600
        is_ota = _is_ota(lineup_id_input)
        url = _build_url(lineup_api, headend_api, c3, postal or "", t, chunk_hours, is_ota=is_ota)

        logging.debug(f"Fetching chunk {idx+1}/{len(offsets)} for lineup {lineup_id_input}")
        logging.debug(f"URL: {url}")

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
                                "thumbnail": ch.get("thumbnail"),
                                "events": []
                            }
                        base = channels_map[cid]
                        for ev in ch.get("events", []) or []:
                            base["events"].append(ev)
                    break

                elif r.status_code == 400:
                    logging.error(f"Bad request for lineup {lineup_id_input} (likely invalid headend or postalCode)")
                    break

                elif r.status_code in (429,) or 500 <= r.status_code < 600:
                    wait = min(60, 2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logging.warning(f"Retry {attempt}/{max_retries} in {wait:.1f}s (status {r.status_code})")
                    time.sleep(wait)

                else:
                    logging.error(f"Zap2it returned HTTP {r.status_code} for chunk {idx+1}")
                    break

            except Exception as e:
                wait = min(30, 2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logging.warning(f"Network error: {e} — retrying in {wait:.1f}s")
                time.sleep(wait)

        if delay_seconds > 0 and idx < len(offsets) - 1:
            time.sleep(delay_seconds)

    channels = list(channels_map.values())
    channels.sort(key=lambda c: (str(c.get("callSign") or ""), str(c.get("channelNo") or "")))
    return channels



# ---------------- Time & XML helpers ----------------
def _zap_iso_to_dt(s):
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
    if not dtobj:
        return ""
    if dtobj.tzinfo is None:
        dtobj = dtobj.replace(tzinfo=_dt.timezone.utc)
    return dtobj.strftime("%Y%m%d%H%M%S %z")

def write_xmltv(channels, out_path: Path, include_thumbnails=True):
    tv = ET.Element("tv")
    channels = sorted(channels, key=lambda c: str(c.get("callSign") or "").lower())

    for ch in channels:
        cid = str(ch.get("stationId") or ch.get("channelId") or "")
        ch_el = ET.SubElement(tv, "channel", {"id": cid})

        # Display names
        for val in [
            ch.get("callSign"),
            ch.get("affiliateName"),
            f"{ch.get('callSign')} {ch.get('affiliateName')}" if ch.get("callSign") and ch.get("affiliateName") else None
        ]:
            if val:
                ET.SubElement(ch_el, "display-name").text = str(val)

        # --- Thumbnail handling ---
        thumb = ch.get("thumbnail")
        if thumb and not thumb.startswith("http"):
            thumb = f"https:{thumb}.jpg"

        if include_thumbnails and thumb:
            ET.SubElement(ch_el, "icon", {"src": str(thumb)})

    # Programme/event data
    for ch in channels:
        events = sorted(ch.get("events", []), key=lambda e: e.get("startTime") or "")
        for ev in events:
            start_dt = _zap_iso_to_dt(ev.get("startTime") or ev.get("start"))
            end_dt = _zap_iso_to_dt(ev.get("endTime") or ev.get("end"))
            prog_el = ET.SubElement(tv, "programme", {
                "start": _xmltv_time(start_dt),
                "stop": _xmltv_time(end_dt),
                "channel": str(ch.get("stationId") or ch.get("channelId") or ""),
            })

            title = ev.get("program", {}).get("title") or ev.get("title")
            if title:
                ET.SubElement(prog_el, "title").text = str(title)

            desc = ev.get("program", {}).get("shortDesc") or ev.get("description")
            if desc:
                ET.SubElement(prog_el, "desc").text = str(desc)

    tree = ET.ElementTree(tv)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)


# ---------------- DB Lookup ----------------
def get_lineups_from_column(column: str, value: str, max_lineups: int = None, db_path: str = DB_FILE):
    if not Path(db_path).exists():
        logging.error(f"Database file {db_path} not found")
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    kw = f"%{value.lower()}%"
    cur.execute(f"SELECT DISTINCT lineupid FROM channels_by_country WHERE lower({column}) LIKE ?", (kw,))
    results = [row[0] for row in cur.fetchall()]
    conn.close()
    if max_lineups and len(results) > max_lineups:
        results = results[:max_lineups]
    logging.info(f"Found {len(results)} lineup IDs for {column}='{value}'")
    return results

def get_lineups_from_keyword(keyword: str, max_lineups: int = None, db_path: str = DB_FILE):
    if not Path(db_path).exists():
        logging.error(f"Database file {db_path} not found")
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    kw = f"%{keyword.lower()}%"
    cur.execute("""
        SELECT DISTINCT lineupid FROM channels_by_country
        WHERE lower(lineup_name) LIKE ?
           OR lower(country_name) LIKE ?
           OR lower(country) LIKE ?
           OR lower(station_name) LIKE ?
           OR lower(lineupid) LIKE ?
    """, (kw, kw, kw, kw, kw))
    results = [row[0] for row in cur.fetchall()]
    conn.close()
    if max_lineups and len(results) > max_lineups:
        results = results[:max_lineups]
    logging.info(f"Found {len(results)} lineup IDs for keyword='{keyword}'")
    return results

def filter_test_lineups(lineup_ids):
    """Remove test/sandbox lineups (those containing 'test' or similar)."""
    filtered = []
    for lid in lineup_ids:
        lid_lower = lid.lower()
        if "test" in lid_lower or "sandbox" in lid_lower or "dummy" in lid_lower:
            logging.debug(f"Filtered out test lineup: {lid}")
            continue
        filtered.append(lid)
    return filtered

def filter_broken_lineups(lineup_ids):
    bad_patterns = ["xumotv", "sandbox", "test", "dummy"]
    result = []
    for lid in lineup_ids:
        lid_lower = lid.lower()
        if any(bad in lid_lower for bad in bad_patterns):
            logging.warning(f"Skipping broken lineup: {lid}")
            continue
        result.append(lid)
    return result

# ---------------- Config Loader ----------------
def load_config():
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
        "EPG_DB_PATH": DB_FILE,
        "EPG_THUMBNAILS": True
    }
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                file_cfg = json.load(f)
            config.update({k.upper(): v for k, v in file_cfg.items()})
            logging.info(f"Loaded configuration from {CONFIG_FILE}")
        except Exception as e:
            logging.warning(f"Failed to read {CONFIG_FILE}: {e}")
    for key in config.keys():
        if key in os.environ:
            val = os.environ[key]
            if key in ["EPG_TIMESPAN_DAYS", "EPG_VERBOSE", "EPG_DELAY", "EPG_MAX_LINEUPS", "EPG_RETRY_COUNT"]:
                val = int(val)
            config[key] = val

    # Convert days to hours
    days = int(config.get("EPG_TIMESPAN_DAYS", 1))
    config["EPG_TIMESPAN"] = min(days, 7) * 24

    # Lookup logic
    if not config["EPG_LINEUP_ID"] and config["EPG_LOOKUP_MODE"] and config["EPG_LOOKUP_VALUE"]:
        max_lineups = config.get("EPG_MAX_LINEUPS")
        mode = config["EPG_LOOKUP_MODE"].lower()
        raw_val = config["EPG_LOOKUP_VALUE"]
        lookup_values = [v.strip() for v in raw_val.split(",") if v.strip()]
        db_path = config.get("EPG_DB_PATH", DB_FILE)
        all_lineups = []

        for val in lookup_values:
            if mode == "country_name":
                all_lineups.extend(get_lineups_from_column("country_name", val, max_lineups, db_path))
            elif mode == "country":
                all_lineups.extend(get_lineups_from_column("country", val, max_lineups, db_path))
            elif mode == "station_name":
                all_lineups.extend(get_lineups_from_column("station_name", val, max_lineups, db_path))
            elif mode == "keyword":
                all_lineups.extend(get_lineups_from_keyword(val, max_lineups, db_path))
            else:
                logging.error(f"Unsupported lookup mode: {mode}")


        filtered_lineups = filter_test_lineups(all_lineups)
        filtered_lineups = filter_broken_lineups(filtered_lineups)
        unique_lineups = list(dict.fromkeys(filtered_lineups))


        # Deduplicate and limit to max_lineups
        unique_lineups = list(dict.fromkeys(filtered_lineups))

        if max_lineups and len(unique_lineups) > max_lineups:
            unique_lineups = unique_lineups[:max_lineups]

        if unique_lineups:
            config["EPG_LINEUP_ID"] = ",".join(unique_lineups)
            logging.info(f"Resolved multiple keywords to lineup IDs: {config['EPG_LINEUP_ID']}")


    return config

# ---------------- Runner ----------------
def run_epg(config):
    lineup_ids = [lid.strip() for lid in str(config["EPG_LINEUP_ID"]).split(",") if lid.strip()]
    if not lineup_ids:
        print("[DEBUG] Lookup mode:", config.get("EPG_LOOKUP_MODE"))
        print("[DEBUG] Lookup value:", config.get("EPG_LOOKUP_VALUE"))
        print("[DEBUG] Lineup IDs found:", lineup_ids)

        logging.error("No lineup IDs found. Check your EPG_LOOKUP_MODE / EPG_LOOKUP_VALUE or EPG_LINEUP_ID.")
        sys.exit(1)

    country = config["EPG_COUNTRY"]
    postal = config["EPG_ZIP"]
    timespan = int(config["EPG_TIMESPAN"])
    delay = int(config["EPG_DELAY"])
    retry_count = int(config.get("EPG_RETRY_COUNT", 3))
    output_file = Path(config["EPG_OUTPUT"])

    logging.info(f"Starting EPG fetch for lineups: {', '.join(lineup_ids)}")
    all_channels = []
    for idx, lineup in enumerate(lineup_ids, 1):
        logging.info(f"Fetching lineup {idx}/{len(lineup_ids)}: {lineup}")
        try:
            channels = fetch_grid(
                country=country,
                lineup_id_input=lineup,
                postal=postal or None,
                timespan=timespan,
                delay_seconds=delay,
                max_retries=retry_count
            )
            all_channels.extend(channels)
        except Exception as e:
            logging.error(f"Failed to fetch lineup {lineup}: {e}", exc_info=(config["EPG_VERBOSE"] >= 2))

    if not all_channels:
        logging.error("No channels retrieved. Exiting.")
        sys.exit(1)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_xmltv(all_channels, output_file, include_thumbnails=config.get("EPG_THUMBNAILS", True))
        logging.info(f"EPG successfully written to {output_file}")
    except Exception as e:
        logging.error(f"Failed to write XMLTV: {e}")
        sys.exit(1)

# ---------------- Main ----------------
if __name__ == "__main__":
    config = load_config()
    setup_logging(config.get("EPG_VERBOSE", 1), config.get("EPG_LOG_FILE"))
    with _single_instance_guard():
        run_epg(config)
