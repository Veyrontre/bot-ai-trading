#economic_calendar.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re

class EconCalendarLoader:
    def __init__(self, output_type="df", retries=3, delay=5):
        """
        Loader economic calendar dari TradingEconomics.com
        :param output_type: "df" untuk pandas, "list" untuk list of dict
        :param retries: jumlah retry saat gagal fetch
        :param delay: delay (detik) sebelum retry
        """
        self.base_url = "https://tradingeconomics.com/calendar"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        self.output_type = output_type
        self.retries = retries
        self.delay = delay

    def fetch(self, start_date=None, end_date=None):
        """
        Fetch dan parse event dari TradingEconomics
        :param start_date: datetime.date atau string format 'YYYY-MM-DD' (default: hari ini)
        :param end_date: datetime.date atau string format 'YYYY-MM-DD' (default: 7 hari ke depan)
        """
        if start_date is None:
            start_date = datetime.now().date()
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=7)).date()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        attempt = 0
        while attempt < self.retries:
            try:
                # Format URL dengan parameter tanggal
                url = f"{self.base_url}?d1={start_date.strftime('%Y-%m-%d')}&d2={end_date.strftime('%Y-%m-%d')}"
                print(f"Fetching URL: {url}")
                
                resp = requests.get(url, headers=self.headers, timeout=30)
                resp.raise_for_status()
                
                # Cek apakah halaman terload dengan benar
                if "calendar" not in resp.text.lower():
                    raise Exception("Calendar content not found in response")
                    
                return self._parse_improved(resp.text, start_date, end_date)
            except Exception as e:
                print(f"[Warning] Fetch attempt {attempt+1} failed: {e}")
                attempt += 1
                if attempt < self.retries:
                    time.sleep(self.delay * attempt)  # Exponential backoff
                    
        print("[Error] Failed to fetch TradingEconomics calendar after retries.")
        return pd.DataFrame() if self.output_type == "df" else []

    def _parse_improved(self, html_text, start_date, end_date):
        """
        Parse HTML page dengan metode yang lebih akurat
        """
        soup = BeautifulSoup(html_text, "html.parser")
        
        # Cari semua baris data
        rows = soup.find_all("tr", {"data-eventid": True})
        
        if not rows:
            print("[Warning] No data rows found with data-eventid attribute")
            # Coba metode alternatif
            return self._parse_fallback(soup, start_date, end_date)
        
        events = []
        print(f"Found {len(rows)} data rows with event IDs")
        
        for i, row in enumerate(rows[:20]):  # Batasi untuk debugging
            try:
                # Ekstrak data dari atribut data-eventid dan data-eventdate
                event_id = row.get("data-eventid", "")
                event_date_str = row.get("data-eventdate", "")
                
                # Cari semua sel dalam baris
                cells = row.find_all("td")
                if len(cells) < 8:
                    continue
                
                # Parse waktu
                time_cell = cells[0]
                time_str = time_cell.get_text(strip=True)
                
                # Parse negara/currency
                country_cell = cells[1]
                country = country_cell.get_text(strip=True)
                
                # Parse event name (biasanya di kolom ke-4)
                event_name = ""
                if len(cells) > 4:
                    event_cell = cells[4]
                    event_name = event_cell.get_text(strip=True)
                
                # Parse actual, forecast, previous
                actual = ""
                forecast = ""
                previous = ""
                
                if len(cells) > 5:
                    actual = cells[5].get_text(strip=True)
                if len(cells) > 6:
                    previous = cells[6].get_text(strip=True)
                if len(cells) > 7:
                    forecast = cells[7].get_text(strip=True)
                
                # Parse importance dari class row
                importance = 1
                row_class = row.get("class", [])
                for cls in row_class:
                    if "calendar-event-importance" in cls:
                        # Coba ekstrak angka dari class
                        match = re.search(r'calendar-event-importance-(\d)', cls)
                        if match:
                            importance = int(match.group(1))
                
                # Parse tanggal dan waktu
                event_datetime = None
                try:
                    # Coba parse dari data-eventdate
                    if event_date_str:
                        # Format biasanya: "2026-01-23T12:30:00"
                        event_datetime = datetime.fromisoformat(event_date_str.replace("Z", ""))
                    else:
                        # Fallback: gunakan start_date dan parse waktu
                        if time_str and ":" in time_str:
                            time_obj = datetime.strptime(time_str, "%I:%M %p").time()
                            event_datetime = datetime.combine(start_date, time_obj)
                        else:
                            event_datetime = datetime.combine(start_date, datetime.min.time())
                except Exception as e:
                    print(f"[Warning] Error parsing datetime for row {i}: {e}")
                    event_datetime = datetime.combine(start_date, datetime.min.time())
                
                # Buat event object
                event = {
                    "event_id": event_id,
                    "datetime": event_datetime,
                    "date": event_datetime.date() if event_datetime else start_date,
                    "time": event_datetime.time() if event_datetime else None,
                    "country": country,
                    "currency": self._country_to_currency(country),
                    "event": event_name,
                    "actual": self._clean_value(actual),
                    "forecast": self._clean_value(forecast),
                    "previous": self._clean_value(previous),
                    "importance": importance,
                    "source": "TradingEconomics",
                    "last_updated": datetime.now()
                }
                
                events.append(event)
                
            except Exception as e:
                print(f"[Warning] Error parsing row {i}: {e}")
                continue
        
        print(f"Successfully parsed {len(events)} events")
        
        if self.output_type == "df":
            df = pd.DataFrame(events)
            if not df.empty:
                # Sort by datetime
                df = df.sort_values("datetime").reset_index(drop=True)
            return df
        return events

    
    def _parse_fallback(self, soup, start_date, end_date):
        """
        Fallback parsing dengan filter tanggal yang benar
        """
        print("Using fallback parsing method...")
        
        # Cari tabel calendar
        table = soup.find("table", {"id": "calendar"})
        if not table:
            table = soup.find("table", class_=re.compile("calendar"))
        
        if not table:
            print("[Error] No calendar table found in fallback")
            return pd.DataFrame() if self.output_type == "df" else []
        
        events = []
        date_headers = []
        
        # Cari semua header tanggal (biasanya row dengan th yang berisi tanggal)
        all_rows = table.find_all("tr")
        current_date = None
        
        for row in all_rows:
            try:
                # Cek jika ini header tanggal
                date_header = row.find("th", colspan=True)  # Header dengan colspan
                if date_header:
                    date_text = date_header.get_text(strip=True)
                    # Coba parse tanggal dari header
                    try:
                        # Contoh: "Thursday January 22 2026"
                        current_date = datetime.strptime(date_text, "%A %B %d %Y").date()
                        date_headers.append(current_date)
                        print(f"Found date header: {current_date}")
                        continue
                    except:
                        try:
                            # Format lain: "Jan 23, 2026"
                            current_date = datetime.strptime(date_text, "%b %d, %Y").date()
                            date_headers.append(current_date)
                            print(f"Found date header: {current_date}")
                            continue
                        except:
                            current_date = None
                
                # Skip jika bukan data row atau belum ada tanggal
                if current_date is None:
                    continue
                
                # Hanya proses data untuk tanggal dalam range yang diminta
                if current_date < start_date or current_date > end_date:
                    continue
                
                # Skip baris header
                if row.find("th") and not row.find("td"):
                    continue
                
                cells = row.find_all("td")
                if len(cells) < 6:
                    continue
                
                # Parse data
                time_str = cells[0].get_text(strip=True) if len(cells) > 0 else ""
                country = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                
                # Event di kolom 4
                event_name = ""
                if len(cells) > 4:
                    event_name = cells[4].get_text(strip=True)
                
                # Parse waktu
                event_datetime = None
                try:
                    if time_str and ":" in time_str:
                        # Coba format 12 hour
                        try:
                            time_obj = datetime.strptime(time_str, "%I:%M %p").time()
                        except:
                            # Coba format 24 hour
                            time_obj = datetime.strptime(time_str, "%H:%M").time()
                        event_datetime = datetime.combine(current_date, time_obj)
                    else:
                        event_datetime = datetime.combine(current_date, datetime.min.time())
                except:
                    event_datetime = datetime.combine(current_date, datetime.min.time())
                
                # Parse values
                actual = cells[5].get_text(strip=True) if len(cells) > 5 else ""
                previous = cells[6].get_text(strip=True) if len(cells) > 6 else ""
                forecast = cells[7].get_text(strip=True) if len(cells) > 7 else ""
                
                # Importance inference
                importance = self._infer_importance_from_event(event_name, country)
                
                event = {
                    "datetime": event_datetime,
                    "date": event_datetime.date(),
                    "time": event_datetime.time() if time_str and ":" in time_str else None,
                    "country": country,
                    "currency": self._country_to_currency(country),
                    "event": event_name,
                    "actual": self._clean_value(actual),
                    "forecast": self._clean_value(forecast),
                    "previous": self._clean_value(previous),
                    "importance": importance,
                    "source": "TradingEconomics",
                    "last_updated": datetime.now()
                }
                
                events.append(event)
                
            except Exception as e:
                continue
        
        print(f"Fallback parsed {len(events)} events for date range {start_date} to {end_date}")
        print(f"Found date headers: {date_headers}")
        
        if self.output_type == "df":
            df = pd.DataFrame(events)
            if not df.empty:
                df = df.sort_values("datetime").reset_index(drop=True)
            return df
        return events

    def _clean_value(self, value):
        """
        Bersihkan nilai dari karakter tidak perlu
        """
        if not value:
            return ""
        
        # Hapus karakter khusus
        value = value.replace('®', '').replace('*', '').strip()
        
        # Konversi string kosong atau "-" ke None
        if value in ["", "-", "--", "N/A", "n/a"]:
            return ""
        
        return value

    def _country_to_currency(self, country_code):
        """
        Convert country code to currency code
        """
        mapping = {
            "US": "USD", "United States": "USD",
            "EU": "EUR", "Euro Area": "EUR", "EA": "EUR",
            "JP": "JPY", "Japan": "JPY",
            "GB": "GBP", "United Kingdom": "GBP",
            "CH": "CHF", "Switzerland": "CHF",
            "CA": "CAD", "Canada": "CAD",
            "AU": "AUD", "Australia": "AUD",
            "NZ": "NZD", "New Zealand": "NZD",
            "CN": "CNY", "China": "CNY",
            "IN": "INR", "India": "INR",
            "BR": "BRL", "Brazil": "BRL",
            "RU": "RUB", "Russia": "RUB",
            "TR": "TRY", "Turkey": "TRY",
            "ZA": "ZAR", "South Africa": "ZAR",
            "MX": "MXN", "Mexico": "MXN",
            "KR": "KRW", "South Korea": "KRW",
            "SG": "SGD", "Singapore": "SGD",
            "DE": "EUR", "Germany": "EUR",
            "FR": "EUR", "France": "EUR",
            "IT": "EUR", "Italy": "EUR",
            "ES": "EUR", "Spain": "EUR",
            "ID": "IDR", "Indonesia": "IDR",
            "AR": "ARS", "Argentina": "ARS",
            "SA": "SAR", "Saudi Arabia": "SAR",
        }
        
        return mapping.get(country_code.upper() if country_code else "", country_code)

    def normalize_event(self, raw):
        """
        Convert ke format internal pipeline
        """
        return {
            "time": pd.to_datetime(raw.get("datetime")),
            "currency": raw.get("currency"),
            "event": raw.get("event"),
            "actual": raw.get("actual"),
            "forecast": raw.get("forecast"),
            "previous": raw.get("previous"),
            "importance": raw.get("importance", 1)
        }
    def _infer_importance_from_event(self, event_name, country):
        """
        Infer importance level berdasarkan nama event dan negara
        """
        event_lower = event_name.lower()
        country_upper = country.upper() if country else ""
        
        # High importance events
        high_keywords = [
            'gdp', 'inflation', 'cpi', 'ppi', 'unemployment', 'employment',
            'interest rate', 'central bank', 'fed', 'ecb', 'boj', 'boe',
            'non-farm', 'nfp', 'payrolls', 'rate decision'
        ]
        
        # Medium importance events  
        medium_keywords = [
            'retail sales', 'pmi', 'manufacturing', 'industrial', 'trade balance',
            'consumer confidence', 'business confidence', 'housing', 'home sales',
            'durable goods', 'factory orders'
        ]
        
        # Check for high importance
        for keyword in high_keywords:
            if keyword in event_lower:
                return 3
        
        # Check for medium importance
        for keyword in medium_keywords:
            if keyword in event_lower:
                return 2
        
        # Default low importance
        return 1

# ==========================
# Test yang lebih baik
# ==========================
if __name__ == "__main__":
    loader = EconCalendarLoader()
    
    print("Testing improved web scraping...")
    df = loader.fetch()
    
    if not df.empty:
        print(f"\nTotal events fetched: {len(df)}")
        print("\nFirst 10 events:")
        print(df[['datetime', 'country', 'currency', 'event', 'actual', 'forecast', 'importance']].head(10))
        
        print("\nImportance distribution:")
        print(df['importance'].value_counts().sort_index())
        
        print("\nSample of important events (importance >= 2):")
        important = df[df['importance'] >= 2]
        if not important.empty:
            print(important[['datetime', 'country', 'event', 'actual', 'forecast', 'importance']].head())
        else:
            print("No high importance events found")
        
        # Save untuk inspeksi
        df.to_csv("economic_calendar_improved.csv", index=False)
        print(f"\nSaved to economic_calendar_improved.csv")
        
    else:
        print("No events fetched")