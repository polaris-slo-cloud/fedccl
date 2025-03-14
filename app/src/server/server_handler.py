import os
import pandas as pd
import json

from src.shared.model.site import Site
from src.server.federated_server import FederatedServer

if __name__ == "__main__":
    site_info_path = os.getenv('SITE_INFO_PATH')
    data_path = os.getenv('DATA_PATH')
    server_port = os.getenv('SERVER_PORT')

    site_infos = pd.read_csv(site_info_path)

    sites = []

    for _, row in site_infos.iterrows():
        site = Site(
            site_id=row['site_id'],
            clusters=json.loads(row["clusters"].replace("\'", "\"")),
            lat=float(row['lat']),
            lng=float(row['lng']),
            zip=int(row['zip']),
            country=row['country'],
            kwp=float(row['kwp']),
            weather_data=row['weather_data']
        )
        sites.append(site)
    
    server = FederatedServer(sites, server_port, data_path)

    server.run()

    

