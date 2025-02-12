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
const randomCoordinatesCount = 5000; // Number of random coordinates to generate
const headless = true; // Set to true for headless execution
const port = 3000; // Port where the local server is running

// Fixed coordinates
const fixedCoordinates = [
];

// Bounding box for random coordinates
const latMin = 45.46;
const latMax = 45.48;
const lngMin = 9.18;
const lngMax = 9.22;

// Output directories
const outputDirImages = path.join(__dirname, "dataset/images");
const outputDirMasks = path.join(__dirname, "dataset/masks");

// Ensure output directories exist
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

// Wait for tiles to load reliably
async function waitForTilesToLoad(page) {
  await page.evaluate(() => {
    return new Promise((resolve) => {
      // We'll consider the tiles loaded once no render events have occurred for this many ms. min 10ms to avoid false positives.
      const idleThreshold = 50;
      let lastRenderTime = Date.now();

      // Update the last render time on every render event.
      const onRender = () => {
        lastRenderTime = Date.now();
      };

      // Attach the render event listener.
      window.map.events.add('render', onRender);

      // Function to check if the map has been idle.
      const checkIfIdle = () => {
        // If no render event occurred for at least idleThreshold ms, assume loading is complete.
        if (Date.now() - lastRenderTime > idleThreshold) {
          window.map.events.remove('render', onRender);
          resolve();
        } else {
          // Otherwise, check again on the next animation frame.
          requestAnimationFrame(checkIfIdle);
        }
      };

      // Start checking.
      checkIfIdle();
    });
  });
}



// Wait for the mask visibility update
async function waitForMaskVisibility(page, expectedVisibility) {
  let attempts = 0;
  const maxAttempts = 30; // Wait for up to 30 seconds

  while (attempts < maxAttempts) {
    const isVisible = await page.evaluate(() => window.maskVisible);
    if (isVisible === expectedVisibility) return;

    await page.waitForTimeout(1000); // Wait 1 second before checking again
    attempts++;
  }

  throw new Error("Mask visibility update timeout");
}

// Capture map screenshots
async function captureMap(page, includeGeoJson, coordinates) {
  try {
    log(`[${includeGeoJson ? "MASK" : "BASE"}] Capturing ${coordinates.length} coordinates...`, VERBOSITY.BASIC);

    // Wait for Azure Maps to be ready
    log("[STATUS] Waiting for Azure Maps object...", VERBOSITY.DETAILED);
    await page.waitForFunction(() => window.map && typeof window.map.setCamera === "function");

    // Toggle mask visibility
    log(`[STATUS] ${includeGeoJson ? "Enabling" : "Disabling"} mask...`, VERBOSITY.DETAILED);
    await page.evaluate((shouldShow) => {
      if (window.setMask) {
        window.setMask(shouldShow);
        window.maskVisible = shouldShow;
      }
    }, includeGeoJson);

    // Ensure mask visibility is updated
    log("[STATUS] Waiting for mask visibility update...", VERBOSITY.DETAILED);
    await waitForMaskVisibility(page, includeGeoJson);

    let counter = 0;
    for (const coords of coordinates) {
      counter++;
      log(`[STATUS] Processing ${counter}/${coordinates.length}: Moving to ${JSON.stringify(coords)}...`, VERBOSITY.BASIC);
      await page.evaluate((c) => window.map.setCamera({ center: [c.lng, c.lat] }), coords);

      log("[STATUS] Waiting for tiles to load...", VERBOSITY.DETAILED);
      await waitForTilesToLoad(page);

      // Capture screenshot
      log("[STATUS] Capturing screenshot...", VERBOSITY.DETAILED);
      const mapElement = await page.locator("#map");
      const fileName = `${coords.lat.toFixed(5)}_${coords.lng.toFixed(5)}.png`;
      const outputPath = includeGeoJson ? path.join(outputDirMasks, fileName) : path.join(outputDirImages, fileName);

      await mapElement.screenshot({
        path: outputPath,
        captureBeyondViewport: false,
        animations: "disabled",
      });

      log(`[SUCCESS] Image saved to: ${outputPath}`, VERBOSITY.BASIC);
    }
  } catch (error) {
    log(`[ERROR] Capture failed: ${error.message}`, VERBOSITY.BASIC);
    throw error;
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
    await page.goto(`http://localhost:${port}/map.html`, { waitUntil: "networkidle" });

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

    // Process base captures
    log("[PROCESSING] Capturing base images...", VERBOSITY.BASIC);
    await captureMap(page, false, allCoordinates);

    // Process mask captures
    log("[PROCESSING] Capturing mask images...", VERBOSITY.BASIC);
    await captureMap(page, true, allCoordinates);

    log("[SUCCESS] All captures completed!", VERBOSITY.BASIC);
  } catch (error) {
    log(`[FATAL ERROR] ${error.message}`, VERBOSITY.BASIC);
  } finally {
    if (browser) {
      log("[CLEANUP] Closing browser...", VERBOSITY.BASIC);
      await browser.close();
    }
  }
})();
