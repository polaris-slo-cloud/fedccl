class Site:
    site_id: str
    clusters: dict
    lat: float
    lng: float
    zip: int
    country: str
    kwp: float
    weather_data: bool

    def __init__(self, site_id: str, clusters: dict, lat: float, lng: float, zip: int, country: str, kwp: float, weather_data: bool):
        self.site_id = site_id
        self.clusters = clusters
        self.lat = lat
        self.lng = lng
        self.zip = zip
        self.country = country
        self.kwp = kwp
        self.weather_data = weather_data