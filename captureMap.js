const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

// Verbosity levels
const VERBOSITY = {
  NONE: 0,
  BASIC: 1,
  DETAILED: 2,
};
const verbosityLevel = VERBOSITY.BASIC; // Set desired verbosity level
const randomCoordinatesCount = 100; // Numero di coordinate casuali da generare
const headless = true; // Impostare su true per eseguire in modalità headless

// Coordinate specifiche fornite dall'utente
const fixedCoordinates = [
  { lat: 45.496398017034764, lng: 9.224118614878336 },
  { lat: 45.506786509037966, lng: 9.229450783645406 },
  { lat: 45.496182354998574, lng: 9.22742856500629 },
  { lat: 45.50016410794893, lng: 9.236237413743334 },
  { lat: 45.49344335822781, lng: 9.228427352122873 },
];

// Limiti del quadrato a Milano per generare punti casuali
const latMin = 45.46;
const latMax = 45.48;
const lngMin = 9.18;
const lngMax = 9.22;

// Percorsi delle directory di output
const outputDirImages = path.join(__dirname, "dataset/images");
const outputDirMasks = path.join(__dirname, "dataset/masks");

// Crea le directory di output se non esistono
if (!fs.existsSync(outputDirImages)) {
  fs.mkdirSync(outputDirImages, { recursive: true });
}
if (!fs.existsSync(outputDirMasks)) {
  fs.mkdirSync(outputDirMasks, { recursive: true });
}

// Logging helper
function log(message, level = VERBOSITY.BASIC) {
  if (verbosityLevel >= level) {
    console.log(message);
  }
}

// Generate random coordinates
function generateRandomCoordinate() {
  const lat = Math.random() * (latMax - latMin) + latMin;
  const lng = Math.random() * (lngMax - lngMin) + lngMin;
  return { lat, lng };
}

// Capture map screenshots
async function captureMap(page, includeGeoJson, coordinates) {
  try {
    log(
      `[${includeGeoJson ? "MASK" : "BASE"}] Starting capture for ${coordinates.length} coordinates...`,
      VERBOSITY.BASIC
    );

    // Wait for map to be ready
    log("[STATUS] Waiting for map object...", VERBOSITY.DETAILED);
    await page.waitForFunction(
      () => window.map && typeof map.setCenter === "function"
    );

    // Toggle GeoJSON visibility
    log(
      `[STATUS] ${includeGeoJson ? "Enabling" : "Disabling"} mask...`,
      VERBOSITY.DETAILED
    );
    await page.evaluate((shouldShow) => setMask(shouldShow), includeGeoJson);

    // Wait for GeoJSON visibility update
    log("[STATUS] Waiting for mask visibility update...", VERBOSITY.DETAILED);
    await page
      .waitForFunction(
        (expectedVisibility) => {
          const style = map.data.getStyle();
          return style.visible === expectedVisibility;
        },
        includeGeoJson,
        { timeout: 10000 }
      )
      .catch((error) =>
        log(`[WARNING] Mask visibility update timeout: ${error.message}`, VERBOSITY.BASIC)
      );

    for (const coords of coordinates) {
      log(
        `[STATUS] Setting map center to: ${JSON.stringify(coords)}...`,
        VERBOSITY.DETAILED
      );
      await page.evaluate((c) => map.setCenter(c), coords);

      log("[STATUS] Waiting for tiles to load...", VERBOSITY.DETAILED);
      await page.evaluate(async () => {
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            reject(new Error("Tiles failed to load within 10 seconds"));
          }, 10000); // 10-second timeout

          google.maps.event.addListenerOnce(map, "tilesloaded", () => {
            clearTimeout(timeout);
            resolve();
          });
        });
      });

      // Capture screenshot
      log("[STATUS] Capturing screenshot...", VERBOSITY.DETAILED);
      const mapElement = await page.locator("#map");
      const fileName = `${coords.lat.toFixed(5)}_${coords.lng.toFixed(5)}.png`;
      const outputPath = includeGeoJson
        ? path.join(outputDirMasks, fileName)
        : path.join(outputDirImages, fileName);

      await mapElement.screenshot({
        path: outputPath,
        captureBeyondViewport: false,
        animations: "disabled",
      });
      log(`[SUCCESS] Image saved to: ${outputPath}`, VERBOSITY.BASIC);
    }
  } catch (error) {
    log(`[ERROR] Capture failed: ${error.message}`, VERBOSITY.BASIC);
    throw error; // Rethrow to bubble up the error
  }
}

// Main execution
(async () => {
  let browser;
  try {
    log("[INIT] Launching browser...", VERBOSITY.BASIC);
    browser = await chromium.launch({ headless: headless });
    const page = await browser.newPage();

    log("[NAVIGATION] Loading page...", VERBOSITY.BASIC);
    await page.goto("http://localhost:3000/map.html", {
      waitUntil: "networkidle",
      timeout: 60000,
    });

    log("[SETUP] Configuring viewport...", VERBOSITY.DETAILED);
    await page.setViewportSize({ width: 600, height: 600 });

    log("[SETUP] Hiding buttons...", VERBOSITY.DETAILED);
    await page.evaluate(() => {
      document.querySelectorAll("button").forEach((button) => {
        button.style.display = "none";
      });
    });

    log("[SETUP] Waiting for map container...", VERBOSITY.DETAILED);
    await page.waitForSelector("#map", { state: "visible", timeout: 15000 });

    // Generate all coordinates
    const allCoordinates = fixedCoordinates.concat(
      Array.from({ length: randomCoordinatesCount }, generateRandomCoordinate)
    );

    // Process base coordinates
    log("[PROCESSING] Starting base captures...", VERBOSITY.BASIC);
    await captureMap(page, false, allCoordinates);

    // Process mask coordinates
    log("[PROCESSING] Starting mask captures...", VERBOSITY.BASIC);
    await captureMap(page, true, allCoordinates);

    log("[SUCCESS] All captures completed!", VERBOSITY.BASIC);
  } catch (error) {
    log(`[FATAL ERROR] Main execution failed: ${error.message}`, VERBOSITY.BASIC);
  } finally {
    if (browser) {
      log("[CLEANUP] Closing browser...", VERBOSITY.BASIC);
      await browser.close();
    }
  }
})();