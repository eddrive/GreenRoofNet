const { chromium, webkit, firefox } = require('playwright');
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

function generateRandomCoordinate() {
    const lat = Math.random() * (latMax - latMin) + latMin;
    const lng = Math.random() * (lngMax - lngMin) + lngMin;
    return { lat, lng };
}

async function captureMap(page, includeGeoJson, coordinates) {
    try {
        console.log(`[${includeGeoJson ? 'MASK' : 'BASE'}] Starting capture for coordinates: ${JSON.stringify(coordinates)}`);

        // Wait for map to be ready
        console.log('[STATUS] Waiting for map object...');
        await page.waitForFunction(() => window.map && typeof map.setCenter === 'function');

        // Set center coordinates
        console.log('[STATUS] Setting map center...');
        await page.evaluate(coords => map.setCenter(coords), coordinates);
        
        // Wait for map to stabilize after movement
        console.log('[STATUS] Waiting for map to stabilize...');
        await page.evaluate(async () => {
            await new Promise(resolve => {
                google.maps.event.addListenerOnce(map, 'idle', resolve);
            });
        });

        // Toggle GeoJSON visibility
        console.log(`[STATUS] ${includeGeoJson ? 'Enabling' : 'Disabling'} mask...`);
        await page.evaluate((shouldShow) => {
            setMask(shouldShow);
        }, includeGeoJson);

        // Wait for GeoJSON visibility update
        console.log('[STATUS] Waiting for mask visibility update...');
        await page.waitForFunction((expectedVisibility) => {
            const style = map.data.getStyle();
            return style.visible === expectedVisibility;
        }, includeGeoJson, { timeout: 10000 });

        // Extra safety delay
        await page.waitForTimeout(500);

        // Capture screenshot
        console.log('[STATUS] Capturing screenshot...');
        const mapElement = await page.locator('#map');
        const fileName = `${coordinates.lat.toFixed(5)}_${coordinates.lng.toFixed(5)}.png`;
        const outputPath = includeGeoJson 
            ? path.join(outputDirMasks, fileName)
            : path.join(outputDirImages, fileName);

        await mapElement.screenshot({ 
            path: outputPath, 
            captureBeyondViewport: false,
            animations: 'disabled'  // Disable CSS animations
        });
        console.log(`[SUCCESS] Image saved to: ${outputPath}`);

    } catch (error) {
        console.error(`[ERROR] Capture failed: ${error.message}`);
        throw error; // Rethrow to bubble up the error
    }
}

(async () => {
    let browser;
    try {
        console.log('[INIT] Launching browser...');
        
        //INFO:  set headless to false to see the browser in action
        browser = await chromium.launch({ headless: false });
        const page = await browser.newPage();
        
        console.log('[NAVIGATION] Loading page...');
        await page.goto('http://localhost:3000/map.html', { 
            waitUntil: 'networkidle',
            timeout: 60000
        });

        console.log('[SETUP] Configuring viewport...');
        await page.setViewportSize({ width: 600, height: 600 });

        console.log('[SETUP] Hiding buttons...');
        await page.evaluate(() => {
            document.querySelectorAll('button').forEach(button => {
                button.style.display = 'none';
            });
        });

        console.log('[SETUP] Waiting for map container...');
        await page.waitForSelector('#map', { state: 'visible', timeout: 15000 });

        // Process fixed coordinates
        console.log('[PROCESSING] Starting fixed coordinates...');
        for (const coords of fixedCoordinates) {
            console.log(`[PROCESSING] Fixed coordinate: ${JSON.stringify(coords)}`);
            await captureMap(page, false, coords);
            await captureMap(page, true, coords);
        }

        // Process random coordinates
        console.log('[PROCESSING] Starting random coordinates...');
        for (let i = 0; i < 2; i++) {
            const coords = generateRandomCoordinate();
            console.log(`[PROCESSING] Random coordinate ${i+1}: ${JSON.stringify(coords)}`);
            await captureMap(page, false, coords);
            await captureMap(page, true, coords);
        }

        console.log('[SUCCESS] All captures completed!');
    } catch (error) {
        console.error(`[FATAL ERROR] Main execution failed: ${error.message}`);
    } finally {
        if (browser) {
            console.log('[CLEANUP] Closing browser...');
            await browser.close();
        }
    }
})();