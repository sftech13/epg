# EPG Fetcher (epg2.py)

Fetches TV Electronic Program Guide data from multiple sources and outputs XMLTV format.

**Version: 2.3.0**

## Supported Sources

| Source | Regions | Notes |
|--------|---------|-------|
| **GraceNote API** | USA, Canada, Latin America, Caribbean | OTA + Cable lineups |
| **globetvapp/epg** | Europe (28 countries) | UK, Germany, France, etc. |
| **i.mjh.nz** | Streaming Services | Pluto TV, Samsung TV+, Plex, Stirr, Roku |

## Quick Start

```bash
# Generate sample config
python3 epg2.py --init-config

# Preview what will be fetched
python3 epg2.py --dry-run --profiles

# Run all profiles
python3 epg2.py --profiles

# Update the lineup database
python3 epg2.py --update-db
```

## Configuration (epg_config.json)

### Global Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `EPG_TIMESPAN_DAYS` | Days of guide data (1-7) | `1` |
| `EPG_PARALLEL` | Enable parallel fetching | `false` |
| `EPG_MAX_WORKERS` | Parallel worker threads | `3` |
| `EPG_DELAY` | Seconds between API requests | `0` |
| `EPG_RETRY_COUNT` | Retries for failed requests | `3` |
| `EPG_THUMBNAILS` | Include channel/program images | `true` |
| `EPG_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `"INFO"` |
| `EPG_LOG_FILE` | Log file path | `""` |
| `EPG_DB_PATH` | SQLite database path | `"zap2it.db"` |

### Profiles (EPG_PROFILES)

Each profile generates a separate output file. Example:

```json
{
  "EPG_TIMESPAN_DAYS": 5,
  "EPG_PARALLEL": true,
  "EPG_MAX_WORKERS": 5,

  "EPG_PROFILES": [
    {
      "name": "US OTA",
      "output": "output/EPG-us-ota.xml.gz",
      "lookup_mode": "ota",
      "lookup_value": "United States",
      "ota_only": true,
      "max_lineups": 50
    },
    {
      "name": "US All Markets",
      "output": "output/EPG-us-all-markets.xml.gz",
      "lookup_mode": "keyword",
      "lookup_value": "new york, los angeles, chicago, houston...",
      "ota_only": true,
      "max_lineups": 200
    },
    {
      "name": "Europe",
      "output": "output/EPG-europe.xml.gz",
      "source": "globetvapp",
      "globetvapp_countries": "uk, germany, france, spain, italy"
    },
    {
      "name": "Streaming",
      "output": "output/EPG-streaming.xml.gz",
      "source": "mjh",
      "mjh_services": "pluto_us, samsung_us, plex_all, stirr, roku"
    }
  ]
}
```

### Profile Options

| Option | Description | Example |
|--------|-------------|---------|
| `name` | Display name | `"US OTA"` |
| `output` | Output file (use `.xml.gz` for compression) | `"output/EPG.xml.gz"` |
| `lookup_mode` | `ota`, `keyword`, `country_name`, `country`, `station_name` | `"keyword"` |
| `lookup_value` | Search terms (comma-separated) | `"california, los angeles"` |
| `ota_only` | Only OTA channels (US only) | `true` |
| `max_lineups` | Limit lineups fetched | `50` |
| `source` | External source: `globetvapp`, `mjh`, `external` | `"globetvapp"` |

---

## Database Update (--update-db)

Discover new lineups and validate/clean existing ones.

```bash
# Full update - discover + validate existing (uses parallel workers from config)
python3 epg2.py --update-db

# Fast update - skip validation of existing lineups
python3 epg2.py --update-db --skip-validation

# Update specific countries only
python3 epg2.py --update-db --update-countries USA,CAN,MEX

# Use custom postal codes file (CSV: country_code,postal_code)
python3 epg2.py --update-db --postal-codes-file my_postcodes.csv
```

### What --update-db does:
1. **Discovers** OTA lineups by testing postal codes against GraceNote API
2. **Fetches** channel lists and adds them to the database
3. **Validates** existing lineups (parallel, with progress/ETA)
4. **Removes** broken lineups that no longer work

### Supported Regions (GraceNote):

| Region | OTA | Cable | Notes |
|--------|-----|-------|-------|
| **USA** | Yes | Yes | Full support, all 50 states |
| **Canada** | No | Yes | Cable lineups only |
| **Mexico** | Yes | Yes | Major cities |
| **Latin America** | Limited | Yes | Chile, Colombia, Peru, Ecuador, Venezuela, Costa Rica |
| **Caribbean** | Limited | Yes | Puerto Rico, Jamaica, Bahamas, Bermuda, Dominican Republic |
| **Europe** | No | No | Use `globetvapp` source instead |

---

## External Sources

### Europe (globetvapp)

```json
{
  "name": "Europe",
  "output": "output/EPG-europe.xml.gz",
  "source": "globetvapp",
  "globetvapp_countries": "uk, germany, france, spain, italy, netherlands"
}
```

**Available countries:** austria, belgium, bulgaria, croatia, czech, denmark, finland, france, germany, greece, hungary, ireland, italy, netherlands, norway, poland, portugal, romania, russia, serbia, slovakia, slovenia, spain, sweden, switzerland, turkey, ukraine, uk

### Streaming Services (i.mjh.nz)

```json
{
  "name": "Streaming",
  "output": "output/EPG-streaming.xml.gz",
  "source": "mjh",
  "mjh_services": "pluto_us, pluto_uk, samsung_us, plex_all, stirr, roku"
}
```

**Available services:**
- **Pluto TV:** pluto_us, pluto_uk, pluto_ca, pluto_de, pluto_fr, pluto_es, pluto_it, pluto_mx, pluto_br, pluto_all
- **Samsung TV+:** samsung_us, samsung_uk, samsung_ca, samsung_de, samsung_fr, samsung_es, samsung_it, samsung_all
- **Plex:** plex_us, plex_uk, plex_ca, plex_mx, plex_all
- **Others:** stirr, roku

### Direct URLs

```json
{
  "name": "Custom",
  "output": "output/EPG-custom.xml.gz",
  "source": "external",
  "urls": "https://example.com/epg1.xml, https://example.com/epg2.xml.gz"
}
```

---

## Command-Line Options

```bash
# Lineup selection
--lineup-id USA-CA90210-X    # Specific lineup ID
--lookup-mode keyword        # ota, keyword, country_name, country, station_name
--lookup-value "california"  # Search term(s)
--ota-only                   # Only OTA channels

# Fetching
--days 3                     # Days of data (1-7)
--parallel                   # Enable parallel fetching
--max-workers 5              # Worker threads

# Output
--output EPG.xml             # Output file
--no-thumbnails              # Exclude images

# Modes
--profiles                   # Run all profiles from config
--dry-run                    # Preview without fetching
--init-config                # Generate sample config

# Database
--update-db                  # Update lineup database
--skip-validation            # Skip validating existing lineups
--update-countries USA,CAN   # Limit to specific countries

# Logging
--log-level DEBUG            # DEBUG, INFO, WARNING, ERROR
--log-file epg.log           # Log to file
```

Run `python3 epg2.py --help` for all options.

---

## Output Compatibility

XMLTV files work with:
- Plex DVR
- Jellyfin
- Emby
- TVHeadend
- xTeVe
- Most IPTV players

---

## Changelog

### v2.3.0
- **Database update feature** - `--update-db` discovers and validates lineups
- Parallel validation with progress/ETA display
- Removes broken lineups automatically
- Uses `EPG_PARALLEL` and `EPG_MAX_WORKERS` from config
- Fixed streaming service URLs (i.mjh.nz)

### v2.2.0
- **External EPG sources** - Europe and streaming services
- Added `globetvapp` source (28 European countries)
- Added `mjh` source (Pluto, Samsung TV+, Plex, Stirr, Roku)
- Gzip compression support

### v2.1.0
- **Multi-profile mode** - multiple outputs per run
- Added `--profiles` flag

### v2.0.0
- Added `--dry-run`, `--init-config`, `--version`
- Progress indicators and summary tables
- Parallel fetching support
