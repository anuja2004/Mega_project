const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

let mainWindow;
const LOG_FILE = path.join(__dirname, "../training_log.json");

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 750,
    backgroundColor: "#0d1117",
    webPreferences: {
      preload: path.join(__dirname, "renderer.js"),
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "index.html"));
}

app.whenReady().then(createWindow);

// Handle training start event from frontend
ipcMain.on("start-training", () => {
  console.log("Starting training...");
  const python = spawn("python", ["../server.py"]);

  python.stdout.on("data", (data) => {
    console.log(`PYTHON: ${data}`);
  });

  python.stderr.on("data", (data) => {
    console.error(`PYTHON ERROR: ${data}`);
  });

  python.on("close", (code) => {
    console.log(`Python process exited with code ${code}`);
  });
});

// Watch for updates to training_log.json
fs.watchFile(LOG_FILE, (curr, prev) => {
  try {
    const content = fs.readFileSync(LOG_FILE, "utf8");
    const data = JSON.parse(content);
    mainWindow.webContents.send("log-updated", data);
  } catch (err) {
    console.error("Error reading log file:", err);
  }
});
