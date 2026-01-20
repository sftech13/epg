# EPG Fetcher (epg2.py)

Fetches TV Electronic Program Guide data from the GraceNote API and outputs XMLTV format.

## Quick Start

```bash
# Generate a sample config file
python3 epg2.py --init-config

# Preview what will be fetched (no API calls)
python3 epg2.py --dry-run

# Run the fetcher
python3 epg2.py
```

## Configuration

Edit `epg_config.json` to customize. All settings can also be set via command-line or environment variables.

### Lineup Selection

| Setting | Description | Example |
|---------|-------------|---------|
| `EPG_LOOKUP_MODE` | How to find lineups: `keyword`, `country`, `country_name`, `station_name`, `ota` | `"keyword"` |
| `EPG_LOOKUP_VALUE` | Comma-separated search terms | `"california, los angeles"` |
| `EPG_OTA_ONLY` | Only fetch over-the-air (antenna) channels | `true` or `false` |
| `EPG_MAX_LINEUPS` | Limit number of lineups to fetch | `5` |
| `EPG_LINEUP_ID` | Specific lineup ID(s) - bypasses lookup | `"USA-CA90210-X"` |
| `EPG_ZIP` | ZIP/postal code (auto-detected if blank) | `"90210"` |

### Fetching Options

| Setting | Description | Default |
|---------|-------------|---------|
| `EPG_TIMESPAN_DAYS` | Days of guide data (1-7) | `1` |
| `EPG_DELAY` | Seconds between API requests | `0` |
| `EPG_RETRY_COUNT` | Retries for failed requests | `3` |
| `EPG_PARALLEL` | Fetch multiple lineups simultaneously | `false` |
| `EPG_MAX_WORKERS` | Parallel worker threads | `3` |

### Output

| Setting | Description | Default |
|---------|-------------|---------|
| `EPG_OUTPUT` | Output XML file path | `"EPG.xml"` |
| `EPG_THUMBNAILS` | Include channel/program images | `true` |
| `EPG_LOG_LEVEL` | Logging: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `"INFO"` |
| `EPG_LOG_FILE` | Log file path (blank = console only) | `""` |

### Database

| Setting | Description | Default |
|---------|-------------|---------|
| `EPG_DB_PATH` | Path to zap2it.db SQLite database | `"zap2it.db"` |

## Command-Line Usage

```bash
# Fetch specific lineup
python3 epg2.py --lineup-id USA-CA90210-X --zip 90210

# Fetch all US OTA channels
python3 epg2.py --lookup-mode ota --lookup-value "United States" --ota-only

# Fetch 3 days, parallel mode
python3 epg2.py --days 3 --parallel --max-workers 5

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

The script generates an XMLTV file compatible with:
- Plex DVR
- Jellyfin
- TVHeadend
- Emby
- Most IPTV players

## Version

Current: **2.0.0**

## Recent Changes (v2.0.0)

- Added `--dry-run` to preview without API calls
- Added `--init-config` to generate sample config
- Added `--version` flag
- Added progress indicators `[1/5]` during fetch
- Added summary table at completion
- Fixed hardcoded paths (now relative to script)
- Added SQL injection protection
- Added `--fallback-zip` option
- Improved `--help` output with defaults shown
