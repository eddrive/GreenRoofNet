import geopandas as gpd
from shapely.geometry import Point
from PIL import Image, ImageDraw
import requests
import random
import io
import os
import math

# Configurazione delle directory di output
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)

# Percorso del file GeoJSON
geojson_path = "./potentialGR.geojson"
gdf = gpd.read_file(geojson_path)

# Parametri immagine satellitare
API_KEY = "***REMOVED***"  # Sostituisci con la tua API key
zoom = 20
size = "640x640"
maptype = "satellite"

# Genera un punto casuale all'interno del bounding box
def get_random_point_within(gdf):
    polygon = gdf.unary_union  # Combina tutte le geometrie
    minx, miny, maxx, maxy = polygon.bounds

    while True:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):
            return random_point

# Scarica immagine satellitare
def download_satellite_image(latitude, longitude, zoom, size, maptype, api_key):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size}&maptype={maptype}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"Errore nel download dell'immagine satellitare: {response.status_code}")

# Genera maschera per le geometrie filtrate
def create_mask(filtered_gdf, latitude, longitude, zoom, size):
    mask = Image.new("L", (640, 640), 0)  # Immagine in scala di grigi, inizialmente vuota
    draw = ImageDraw.Draw(mask)

    def latlon_to_pixels(lat, lon, center_lat, center_lon, zoom, img_size):
        scale = 2 ** zoom * 256  # Scala per il livello di zoom e dimensione base (256px)

        def lat_to_y(lat):
            siny = math.sin(math.radians(lat))
            y = 0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)
            return y * scale

        x = (lon + 180) / 360 * scale
        y = lat_to_y(lat)
        center_x = (center_lon + 180) / 360 * scale
        center_y = lat_to_y(center_lat)
        pixel_x = int((x - center_x + img_size[0] / 2))
        pixel_y = int((y - center_y + img_size[1] / 2))
        return (pixel_x, pixel_y)

    center_lat, center_lon = latitude, longitude
    img_size = (640, 640)

    for _, row in filtered_gdf.iterrows():
        geom = row.geometry
        if geom.is_empty:
            print("Geometria vuota, saltata.")
            continue

        if geom.geom_type == "Polygon":
            coords = [
                latlon_to_pixels(lat, lon, center_lat, center_lon, zoom, img_size)
                for lon, lat, *_ in geom.exterior.coords
            ]
            print(f"Coordinate pixel per Polygon: {coords}")
            draw.polygon(coords, fill=1)
        elif geom.geom_type == "MultiPolygon":
            for polygon in geom.geoms:
                coords = [
                    latlon_to_pixels(lat, lon, center_lat, center_lon, zoom, img_size)
                    for lon, lat, *_ in polygon.exterior.coords
                ]
                draw.polygon(coords, fill=1)

        # Salva la maschera parziale per il debug
        mask.save(f"dataset/masks/mask_debug_partial.png")

    return mask

# Pipeline per creare immagine e maschera
def generate_dataset_image(index, gdf, zoom, size, maptype, api_key):
    random_point = get_random_point_within(gdf)
    latitude, longitude = random_point.y, random_point.x
    print(f"Punto casuale generato: {latitude}, {longitude}")

    satellite_image = download_satellite_image(latitude, longitude, zoom, size, maptype, api_key)
    satellite_image_path = f"dataset/images/image_{index:03d}.png"
    satellite_image.save(satellite_image_path)

    gdf_projected = gdf.to_crs("EPSG:32632")
    center_point_projected = gpd.GeoSeries([random_point], crs="EPSG:4326").to_crs("EPSG:32632").iloc[0]

    max_distance_meters = 500
    gdf_projected['distance_to_center'] = gdf_projected.geometry.distance(center_point_projected)
    filtered_gdf = gdf_projected[gdf_projected['distance_to_center'] <= max_distance_meters]

    print(f"Geometrie filtrate: {len(filtered_gdf)}")

    if filtered_gdf.empty:
        print("Nessuna geometria filtrata, maschera vuota.")
        return

    mask = create_mask(filtered_gdf, latitude, longitude, zoom, size)
    mask_path = f"dataset/masks/mask_{index:03d}.png"
    mask.save(mask_path)
    print(f"Immagine e maschera salvate: {satellite_image_path}, {mask_path}")

# Genera un dataset con N immagini
def create_dataset(n_images, gdf, zoom, size, maptype, api_key):
    for i in range(n_images):
        try:
            generate_dataset_image(i, gdf, zoom, size, maptype, api_key)
        except Exception as e:
            print(f"Errore nella generazione dell'immagine {i}: {e}")

# Esegui il processo di generazione
n_images = 3
create_dataset(n_images, gdf, zoom, size, maptype, API_KEY)
