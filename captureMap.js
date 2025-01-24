const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// Coordinate specifiche fornite dall'utente
const fixedCoordinates = [
    { lat: 45.496398017034764, lng: 9.224118614878336 },
    { lat: 45.506786509037966, lng: 9.229450783645406 },
    { lat: 45.496182354998574, lng: 9.22742856500629 },
    { lat: 45.50016410794893, lng: 9.236237413743334 },
    { lat: 45.49344335822781, lng: 9.228427352122873 }
];

// Limiti del quadrato a Milano per generare punti casuali
const latMin = 45.4600;
const latMax = 45.4800;
const lngMin = 9.1800;
const lngMax = 9.2200;

// Percorsi delle directory di output
const outputDirImages = path.join(__dirname, 'dataset/images');
const outputDirMasks = path.join(__dirname, 'dataset/masks');

// Crea le directory di output se non esistono
if (!fs.existsSync(outputDirImages)) {
    fs.mkdirSync(outputDirImages, { recursive: true });
}
if (!fs.existsSync(outputDirMasks)) {
    fs.mkdirSync(outputDirMasks, { recursive: true });
}

// Funzione per generare un punto casuale all'interno del quadrato
function generateRandomCoordinate() {
    const lat = Math.random() * (latMax - latMin) + latMin;
    const lng = Math.random() * (lngMax - lngMin) + lngMin;
    return { lat, lng };
}

async function captureMap(includeGeoJson, coordinates) {
    console.log(`Inizio cattura immagine. GeoJSON incluso: ${includeGeoJson}, Coordinate: ${JSON.stringify(coordinates)}`);

    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    // Carica la pagina HTML locale
    await page.goto('http://localhost:5500/map.html');
    await page.setViewport({ width: 600, height: 600 });

    // Nascondi tutti i bottoni e controlli
    await page.evaluate(() => {
        document.querySelectorAll('button').forEach((button) => {
            button.style.display = 'none';
        });
    });

    // Sposta il centro della mappa sulla coordinata specificata
    await page.evaluate((coords) => {
        map.setCenter(coords);
    }, coordinates);

    // Configura la visibilità e il colore del GeoJSON
    if (includeGeoJson) {
        console.log('Abilitando maschera GeoJSON blu');
        await page.evaluate(() => {
            updateGeoJsonStyle("blue", "blue", true); // Maschera blu
        });
    } else {
        console.log('Disabilitando maschera GeoJSON');
        await page.evaluate(() => {
            updateGeoJsonStyle("transparent", "transparent", false); // Nessuna maschera
        });
    }

    // Attendi il rendering completo della mappa
    await new Promise((resolve) => setTimeout(resolve, 20000));

    // Cattura l'elemento della mappa
    const mapElement = await page.$('#map');
    if (!mapElement) {
        throw new Error('Elemento mappa non trovato');
    }

    // Genera il nome del file in base alle coordinate
    const fileName = `${coordinates.lat.toFixed(5)}_${coordinates.lng.toFixed(5)}.png`;
    const outputPath = includeGeoJson
        ? path.join(outputDirMasks, fileName)
        : path.join(outputDirImages, fileName);

    // Salva lo screenshot
    await mapElement.screenshot({ path: outputPath });
    console.log(`Immagine salvata in: ${outputPath}`);

    await browser.close();
}

// Esegui screenshot con le coordinate fisse e casuali
(async () => {
    // Prima cattura le immagini per le coordinate fisse
    for (const coords of fixedCoordinates) {
        await captureMap(false, coords); // Immagine normale
        await captureMap(true, coords);  // Maschera GeoJSON
    }

    // Poi cattura le immagini per due coordinate casuali
    const randomPoints = 2;
    for (let i = 0; i < randomPoints; i++) {
        const coords = generateRandomCoordinate();
        await captureMap(false, coords); // Immagine normale
        await captureMap(true, coords);  // Maschera GeoJSON
    }
})();
