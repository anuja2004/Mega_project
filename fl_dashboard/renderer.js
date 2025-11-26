const { ipcRenderer } = require("electron");

let accuracyChart, lossChart;

// Initialize Charts
function initCharts() {
  const ctxAcc = document.getElementById("accuracyChart").getContext("2d");
  const ctxLoss = document.getElementById("lossChart").getContext("2d");

  accuracyChart = new Chart(ctxAcc, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Average Accuracy",
        data: [],
        borderColor: "#22c55e",
        tension: 0.3
      }]
    },
    options: { scales: { y: { beginAtZero: true, max: 1 } } }
  });

  lossChart = new Chart(ctxLoss, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Average Loss",
        data: [],
        borderColor: "#ef4444",
        tension: 0.3
      }]
    },
    options: { scales: { y: { beginAtZero: true, max: 1 } } }
  });
}

initCharts();

// Start Training Button
document.getElementById("startBtn").addEventListener("click", () => {
  ipcRenderer.send("start-training");
  document.getElementById("roundSummary").innerText = "Training started...";
});

// Listen for log updates from main.js
ipcRenderer.on("log-updated", (event, data) => {
  const latest = data[data.length - 1];
  const round = latest.round;
  const acc = latest.avg_acc || latest.accuracy;
  const loss = latest.avg_loss || latest.loss;

  document.getElementById("roundSummary").innerHTML =
    `<b>Round ${round}</b><br>Accuracy: ${acc.toFixed(4)} | Loss: ${loss.toFixed(4)}`;

  accuracyChart.data.labels.push(`Round ${round}`);
  accuracyChart.data.datasets[0].data.push(acc);
  accuracyChart.update();

  lossChart.data.labels.push(`Round ${round}`);
  lossChart.data.datasets[0].data.push(loss);
  lossChart.update();
});
