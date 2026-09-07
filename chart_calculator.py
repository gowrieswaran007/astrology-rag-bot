"""
chart_calculator.py
---------------------
Computes a Vedic (sidereal) birth chart from Date of Birth, Time of
Birth, and Place of Birth using SKYFIELD - a pure-Python astronomy
library (no C++ compiler needed, unlike pyswisseph).

Skyfield downloads a small NASA JPL ephemeris file (de421.bsp, ~17MB)
ONCE automatically and caches it in this folder - no manual setup needed.

Geocoding (place name -> latitude/longitude) uses the free, open
Nominatim/OpenStreetMap service via geopy.
Timezone lookup uses timezonefinder (fully offline, open source).

Ayanamsa (the offset between tropical and sidereal zodiac) uses the
standard Lahiri formula - the same reference system used in KP astrology.
"""

import math
from datetime import datetime
from skyfield.api import load
from skyfield.framelib import ecliptic_frame
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# Bodies available in the de421 ephemeris file
PLANET_KEYS = {
    "Sun": "sun",
    "Moon": "moon",
    "Mercury": "mercury",
    "Venus": "venus",
    "Mars": "mars",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
}

geolocator = Nominatim(user_agent="astrology_multiagent_app")
tf = TimezoneFinder()

# Ephemeris + timescale are loaded once and cached to disk (first run
# downloads de421.bsp automatically into this folder)
_loader = load
_eph = _loader('de421.bsp')
_ts = _loader.timescale()
_earth = _eph['earth']


def get_lat_lon(place_name):
    """Convert a place name (e.g. 'Chennai, India') into latitude/longitude."""
    location = geolocator.geocode(place_name, timeout=10)
    if not location:
        raise ValueError(f"Could not find location: '{place_name}'. Try a more specific place name.")
    return location.latitude, location.longitude


def get_timezone(lat, lon):
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if not tz_name:
        raise ValueError("Could not determine timezone for this location.")
    return tz_name


def local_to_utc(dob, tob, tz_name):
    """dob: 'YYYY-MM-DD', tob: 'HH:MM' (24hr) -> returns UTC datetime"""
    naive_dt = datetime.strptime(f"{dob} {tob}", "%Y-%m-%d %H:%M")
    local_tz = pytz.timezone(tz_name)
    local_dt = local_tz.localize(naive_dt)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt


def lahiri_ayanamsa(year):
    """Approximate Lahiri ayanamsa (degrees) for a given year.
    Standard reference value: ~23.85 degrees in year 2000,
    precessing at ~50.29 arcseconds per year."""
    return 23.85 + (year - 2000) * (50.29 / 3600)


def longitude_to_sign(longitude):
    """Convert 0-360 degree ecliptic longitude into zodiac sign + degree within sign."""
    longitude = longitude % 360
    sign_index = int(longitude // 30)
    degree_in_sign = longitude % 30
    return ZODIAC_SIGNS[sign_index], round(degree_in_sign, 2)


def mean_node_longitude(t):
    """Mean lunar node (Rahu) tropical longitude using a standard
    astronomical formula (degrees)."""
    T = (t.tt - 2451545.0) / 36525.0
    node = 125.0445222 - 1934.1362608 * T + 0.0020708 * T**2 + (T**3) / 450000
    return node % 360


def calculate_ascendant(t, lat, lon, ayanamsa):
    """Computes the Ascendant (Lagna) using Local Sidereal Time,
    geographic latitude, and the ecliptic obliquity - the standard
    formula used in astrology software."""
    gmst_hours = t.gmst  # Greenwich Mean Sidereal Time, in hours
    lst_hours = (gmst_hours + lon / 15.0) % 24
    ramc_deg = lst_hours * 15.0  # Right Ascension of Midheaven

    obliquity_deg = 23.4367  # mean obliquity of the ecliptic (good approx for any recent date)

    ramc_rad = math.radians(ramc_deg)
    obliquity_rad = math.radians(obliquity_deg)
    lat_rad = math.radians(lat)

    y = -math.cos(ramc_rad)
    x = (math.sin(ramc_rad) * math.cos(obliquity_rad)
         + math.tan(lat_rad) * math.sin(obliquity_rad))
    asc_tropical = math.degrees(math.atan2(y, x)) % 360

    asc_sidereal = (asc_tropical - ayanamsa) % 360
    return asc_sidereal


def calculate_birth_chart(name, dob, tob, place):
    """
    name: str
    dob:  'YYYY-MM-DD'
    tob:  'HH:MM' (24-hour, local time at birth place)
    place: str, e.g. 'Chennai, India'

    Returns a dict with Ascendant + all planetary positions (sidereal/Vedic).
    """
    lat, lon = get_lat_lon(place)
    tz_name = get_timezone(lat, lon)
    utc_dt = local_to_utc(dob, tob, tz_name)

    t = _ts.utc(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour, utc_dt.minute)
    ayanamsa = lahiri_ayanamsa(utc_dt.year)

    chart = {
        "name": name,
        "dob": dob,
        "tob": tob,
        "place": place,
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "timezone": tz_name,
        "planets": {},
    }

    # Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn
    for planet_name, body_key in PLANET_KEYS.items():
        body = _eph[body_key]
        astrometric = _earth.at(t).observe(body)
        lat_ecl, lon_ecl, dist = astrometric.frame_latlon(ecliptic_frame)
        tropical_long = lon_ecl.degrees
        sidereal_long = (tropical_long - ayanamsa) % 360
        sign, degree = longitude_to_sign(sidereal_long)
        chart["planets"][planet_name] = {"sign": sign, "degree": degree}

    # Rahu (mean lunar node) + Ketu (always exactly opposite Rahu)
    rahu_tropical = mean_node_longitude(t)
    rahu_sidereal = (rahu_tropical - ayanamsa) % 360
    ketu_sidereal = (rahu_sidereal + 180) % 360
    rahu_sign, rahu_deg = longitude_to_sign(rahu_sidereal)
    ketu_sign, ketu_deg = longitude_to_sign(ketu_sidereal)
    chart["planets"]["Rahu"] = {"sign": rahu_sign, "degree": rahu_deg}
    chart["planets"]["Ketu"] = {"sign": ketu_sign, "degree": ketu_deg}

    # Ascendant (Lagna)
    asc_sidereal = calculate_ascendant(t, lat, lon, ayanamsa)
    asc_sign, asc_degree = longitude_to_sign(asc_sidereal)
    chart["ascendant"] = {"sign": asc_sign, "degree": asc_degree}

    return chart


def chart_to_text(chart):
    """Convert the chart dict into a readable text summary for the LLM."""
    lines = [
        f"Birth Chart for {chart['name']}",
        f"Born: {chart['dob']} at {chart['tob']} in {chart['place']}",
        f"Ascendant (Lagna): {chart['ascendant']['sign']} "
        f"({chart['ascendant']['degree']}°)",
        "",
        "Planetary Positions (Sidereal/Vedic, Lahiri Ayanamsa):",
    ]
    for planet, data in chart["planets"].items():
        lines.append(f"  - {planet}: {data['sign']} ({data['degree']}°)")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual test
    test_chart = calculate_birth_chart(
        name="Test User",
        dob="1995-08-15",
        tob="14:30",
        place="Chennai, India",
    )
    print(chart_to_text(test_chart))