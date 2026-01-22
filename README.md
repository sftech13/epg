# EPG Fetcher (epg2.py)

Fetches TV Electronic Program Guide data from the GraceNote API and outputs XMLTV format.

## Quick Start

```bash
# Generate a sample config file
python3 epg2.py --init-config

# Preview what will be fetched (no API calls)
python3 epg2.py --dry-run

# Run single fetch
python3 epg2.py

# Run all profiles (multiple outputs)
python3 epg2.py --profiles
```

## Multi-Profile Mode (NEW!)

Define multiple EPG outputs in one run. Each profile gets its own output file with different filters.

### Example Config with Profiles

```json
{
  "EPG_TIMESPAN_DAYS": 2,
  "EPG_PARALLEL": true,
  "EPG_MAX_WORKERS": 5,

  "EPG_PROFILES": [
    {
      "name": "US OTA",
      "output": "docs/EPG-us-ota.xml",
      "lookup_mode": "ota",
      "lookup_value": "United States",
      "ota_only": true,
      "max_lineups": 50
    },
    {
      "name": "California Cable",
      "output": "docs/EPG-california.xml",
      "lookup_mode": "keyword",
      "lookup_value": "california, los angeles",
      "ota_only": false,
      "max_lineups": 10
    },
    {
      "name": "Canada OTA",
      "output": "docs/EPG-canada-ota.xml",
      "lookup_mode": "ota",
      "lookup_value": "Canada",
      "ota_only": true,
      "max_lineups": 30
    }
  ]
}
```

### Running Profiles

```bash
# Preview all profiles
python3 epg2.py --profiles --dry-run

# Run all profiles
python3 epg2.py --profiles
```

### Profile Options

| Option | Description | Example |
|--------|-------------|---------|
| `name` | Display name for the profile | `"US OTA"` |
| `output` | Output XML file path | `"docs/EPG-us.xml"` |
| `lookup_mode` | How to find lineups | `"ota"`, `"keyword"`, `"country_name"` |
| `lookup_value` | Search value(s), comma-separated | `"United States"` |
| `ota_only` | Only OTA channels | `true` or `false` |
| `max_lineups` | Max lineups to fetch | `50` |

---

## Single Run Mode

If no profiles are defined (or `--profiles` not used), the script runs in single-output mode.

### Configuration

Edit `epg_config.json` to customize. All settings can also be set via command-line or environment variables.

#### Lineup Selection

| Setting | Description | Example |
|---------|-------------|---------|
| `EPG_LOOKUP_MODE` | How to find lineups: `keyword`, `country`, `country_name`, `station_name`, `ota` | `"keyword"` |
| `EPG_LOOKUP_VALUE` | Comma-separated search terms | `"california, los angeles"` |
| `EPG_OTA_ONLY` | Only fetch over-the-air (antenna) channels | `true` or `false` |
| `EPG_MAX_LINEUPS` | Limit number of lineups to fetch | `5` |
| `EPG_LINEUP_ID` | Specific lineup ID(s) - bypasses lookup | `"USA-CA90210-X"` |
| `EPG_ZIP` | ZIP/postal code (auto-detected if blank) | `"90210"` |

#### Fetching Options

| Setting | Description | Default |
|---------|-------------|---------|
| `EPG_TIMESPAN_DAYS` | Days of guide data (1-7) | `1` |
| `EPG_DELAY` | Seconds between API requests | `0` |
| `EPG_RETRY_COUNT` | Retries for failed requests | `3` |
| `EPG_PARALLEL` | Fetch multiple lineups simultaneously | `false` |
| `EPG_MAX_WORKERS` | Parallel worker threads | `3` |

#### Output

| Setting | Description | Default |
|---------|-------------|---------|
| `EPG_OUTPUT` | Output XML file path | `"EPG.xml"` |
| `EPG_THUMBNAILS` | Include channel/program images | `true` |
| `EPG_LOG_LEVEL` | Logging: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `"INFO"` |
| `EPG_LOG_FILE` | Log file path (blank = console only) | `""` |

#### Database

| Setting | Description | Default |
|---------|-------------|---------|
| `EPG_DB_PATH` | Path to zap2it.db SQLite database | `"zap2it.db"` |

---

## Command-Line Usage

```bash
# Fetch specific lineup
python3 epg2.py --lineup-id USA-CA90210-X --zip 90210

# Fetch all US OTA channels
python3 epg2.py --lookup-mode ota --lookup-value "United States" --ota-only

# Fetch 3 days, parallel mode
python3 epg2.py --days 3 --parallel --max-workers 5

# Run all profiles
python3 epg2.py --profiles

# Debug logging
python3 epg2.py --log-level DEBUG
```

Run `python3 epg2.py --help` for all options.

## Environment Variables

All settings can be set as environment variables:

```bash
export EPG_LOOKUP_MODE=keyword
export EPG_LOOKUP_VALUE="new york"
python3 epg2.py
```

## Output

The script generates XMLTV files compatible with:
- Plex DVR
- Jellyfin
- TVHeadend
- Emby
- Most IPTV players

## Available Regions

The GraceNote API database contains 52 countries, but **only some regions actually work** with the API.

### Supported Regions

| Region | Status | Notes |
|--------|--------|-------|
| USA | **Works** | OTA (over-the-air) and cable lineups both work |
| Canada | **Works** | Cable only (OTA not supported by API) |
| Latin America | **Works** | Cable lineups for Chile, Argentina, Colombia, Peru, etc. |
| Europe | **Not supported** | API returns errors for all European lineups |
| UK | **Not supported** | API returns errors for all UK lineups |

### Pre-configured Profiles

| Profile | Countries | Lineups |
|---------|-----------|---------|
| US OTA | United States (OTA) | ~50 |
| California Cable | US cable (CA region) | 10 |
| Canada Cable | Canada | 10 |
| Latin America | CHL, ARG, COL, PER, ECU, VEN, MEX, CRI | 20 |

### Countries in Database (API Support Varies)

**Confirmed Working:**
- USA (3,356 lineups) - OTA + cable supported
- Canada (119 lineups) - cable only
- Chile (48), Colombia (42), Argentina (35), Peru (30), Ecuador (26)
- Venezuela (23), Costa Rica (13), Mexico (15)

**Likely Working (Latin America):**
- Puerto Rico (10), Guatemala (9), Panama (9), Dominican Republic (7)
- Bolivia (6), Paraguay (5), Honduras (5), Uruguay (4)
- El Salvador (4), Nicaragua (4)

**Caribbean (untested):**
- Jamaica (2), Bahamas (1), Trinidad & Tobago (1)
- Cayman Islands (1), Curacao (1), Aruba (1), Belize (2)

**NOT Working (Europe):**
- Poland (17), Germany (15), Switzerland (15), France (12)
- Austria (11), Norway (10), Netherlands (10), Spain (10)
- Denmark (10), Belgium (10), Sweden (9), Ireland (9)
- Finland (8), Italy (7)
- UK (all lineups fail)

### Adding a New Region

To add a new region profile, add an entry to `EPG_PROFILES` in `epg_config.json`:

**Example: Adding Chile**
```json
{
  "name": "Chile",
  "output": "docs/EPG-chile.xml.gz",
  "lookup_mode": "country_name",
  "lookup_value": "Chile",
  "ota_only": false,
  "max_lineups": 20
}
```

**Example: Adding Caribbean**
```json
{
  "name": "Caribbean",
  "output": "docs/EPG-caribbean.xml.gz",
  "lookup_mode": "keyword",
  "lookup_value": "jamaica, bahamas, trinidad, cayman, curacao, aruba, belize",
  "ota_only": false,
  "max_lineups": 15
}
```

**Example: Adding More Latin America**
```json
{
  "name": "Central America",
  "output": "docs/EPG-central-america.xml.gz",
  "lookup_mode": "keyword",
  "lookup_value": "guatemala, honduras, el salvador, nicaragua, panama",
  "ota_only": false,
  "max_lineups": 20
}
```

> **Note:** European lineups (Germany, France, UK, etc.) are in the database but the GraceNote API does not serve data for them. Only use USA, Canada, and Latin American regions.

### Lookup Modes

| Mode | Description | Example Value |
|------|-------------|---------------|
| `country_name` | Match by full country name | `"Chile"`, `"Canada"` |
| `keyword` | Match lineup names containing keyword | `"california, los angeles"` |
| `ota` | US OTA lineups only | `"United States"` |
| `country` | Match by 3-letter country code | `"CHL"`, `"ARG"` |
| `station_name` | Match by station/channel name | `"ESPN"`, `"CNN"` |

### Tips

- Use `.xml.gz` extension for gzip-compressed output (smaller files)
- Set `ota_only: false` for all non-US regions (OTA not supported)
- Use `max_lineups` to limit fetch time and file size
- Test with `--dry-run --profiles` before running
- Multiple keywords can be comma-separated

---

## Version

Current: **2.1.0**

## Changelog

### v2.1.0
- **NEW: Multi-profile mode** - generate multiple EPG files in one run
- Added `--profiles` flag for multi-output mode
- Profiles support different filters per output file

### v2.0.0
- Added `--dry-run` to preview without API calls
- Added `--init-config` to generate sample config
- Added `--version` flag
- Added progress indicators `[1/5]` during fetch
- Added summary table at completion
- Fixed hardcoded paths (now relative to script)
- Added SQL injection protection
- Added `--fallback-zip` option
- Improved `--help` output with defaults shown
