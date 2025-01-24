import json

def clean_geojson(input_path, output_path):
    """
    Rimuove le coordinate di elevazione da un GeoJSON e corregge eventuali errori di sintassi.

    :param input_path: Percorso del file GeoJSON originale.
    :param output_path: Percorso del file GeoJSON pulito.
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Sostituisci eventuali virgole in eccesso nelle proprietà
    content = content.replace(', }', ' }')
    content = content.replace(',]', ']')

    # Carica il contenuto come JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Errore di decodifica JSON: {e}")
        return

    # Funzione ricorsiva per rimuovere il terzo valore nelle coordinate
    def clean_coordinates(coords):
        if isinstance(coords[0][0], list):
            # Caso MultiPolygon o Polygon
            return [clean_coordinates(c) for c in coords]
        else:
            # Caso lista di coordinate [lon, lat, alt]
            return [[lon, lat] for lon, lat, *_ in coords]

    # Array per i centri dei primi 5 poligoni
    polygon_centers = []

    # Itera sulle feature e modifica le geometrie
    for i, feature in enumerate(data['features']):
        if len(polygon_centers) >= 5:
            break

        geometry = feature['geometry']
        if geometry['type'] == 'MultiPolygon':
            geometry['coordinates'] = clean_coordinates(geometry['coordinates'])
            # Calcola il centro del primo poligono del MultiPolygon
            first_polygon = geometry['coordinates'][0]
            all_coords = [coord for ring in first_polygon for coord in ring]
            center_lon = sum(lon for lon, lat in all_coords) / len(all_coords)
            center_lat = sum(lat for lon, lat in all_coords) / len(all_coords)
            polygon_centers.append([center_lon, center_lat])
        elif geometry['type'] == 'Polygon':
            geometry['coordinates'] = clean_coordinates(geometry['coordinates'])
            # Calcola il centro del Polygon
            all_coords = [coord for ring in geometry['coordinates'] for coord in ring]
            center_lon = sum(lon for lon, lat in all_coords) / len(all_coords)
            center_lat = sum(lat for lon, lat in all_coords) / len(all_coords)
            polygon_centers.append([center_lon, center_lat])

    # Salva il GeoJSON pulito
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    # Stampa i centri dei primi 5 poligoni
    print("Centri dei primi 5 poligoni:")
    print(polygon_centers)

# Percorsi dei file
input_geojson = "potentialGR.geojson"
output_geojson = "cleaned_potentialGR.geojson"

# Esegui la pulizia
clean_geojson(input_geojson, output_geojson)

print(f"File pulito salvato in: {output_geojson}")
