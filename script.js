// Regression mit FFNN in TensorFlow.js
window.addEventListener("load", async () => {
  const overlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");

  const N = 100;
  const noiseVar = 0.05;

  // Ziel-Funktion
  const f = x => 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;

  // Gaußsches Rauschen
  function gaussianNoise(mean, stdDev) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) + mean;
  }

  // Daten erzeugen: Math.random zwischen 2 bis -2 wird in ein Array gepackt
  const xData = Array.from({ length: N }, () => Math.random() * 4 - 2);
  // berechne y und packe in eine map
  const yClean = xData.map(f);
  // rauschen hinzugefügt
  const yNoisy = yClean.map(y => y + gaussianNoise(0, Math.sqrt(noiseVar)));

  // Training/Test Split Aufteilung 50/50
  const trainIndices = [...Array(N).keys()].sort(() => 0.5 - Math.random()).slice(0, 50);
  const testIndices = [...Array(N).keys()].filter(i => !trainIndices.includes(i));

  const split = (arr, idx) => idx.map(i => arr[i]);

  const xTrain = split(xData, trainIndices);
  const yTrainClean = split(yClean, trainIndices);
  const yTrainNoisy = split(yNoisy, trainIndices);
  const xTest = split(xData, testIndices);
  const yTestClean = split(yClean, testIndices);
  const yTestNoisy = split(yNoisy, testIndices);

  // Tensor-Konvertierung
  const toTensor = (x, y) => [tf.tensor2d(x, [x.length, 1]), tf.tensor2d(y, [y.length, 1])];

  // Modellaufbau
  const createModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    return model;
  };

  // Training
  const trainModel = async (model, xT, yT, epochs) => {
    await model.fit(xT, yT, {
      epochs,
      batchSize: 32,
      shuffle: true
    });
  };

  // Vorhersage + MSE
  const evaluateModel = async (model, xT, yT) => {
    const pred = model.predict(xT);
    const mse = tf.losses.meanSquaredError(yT, pred).dataSync()[0];
    return [pred.dataSync(), mse];
  };

  // Modelle erstellen
  const modelClean = createModel();
  const modelBest = createModel();
  const modelOver = createModel();

  loadingText.textContent = "📊 Trainiere Modelle...";

  const [xT1, yT1] = toTensor(xTrain, yTrainClean);
  const [xT2, yT2] = toTensor(xTrain, yTrainNoisy);
  const [xTe, yTe1] = toTensor(xTest, yTestClean);
  const [_, yTe2] = toTensor(xTest, yTestNoisy);

  await trainModel(modelClean, xT1, yT1, 100);
  await trainModel(modelBest, xT2, yT2, 50);
  await trainModel(modelOver, xT2, yT2, 300);

  const [yP1, mseTrain1] = await evaluateModel(modelClean, xT1, yT1);
  const [yPt1, mseTest1] = await evaluateModel(modelClean, xTe, yTe1);

  const [yP2, mseTrain2] = await evaluateModel(modelBest, xT2, yT2);
  const [yPt2, mseTest2] = await evaluateModel(modelBest, xTe, yTe2);

  const [yP3, mseTrain3] = await evaluateModel(modelOver, xT2, yT2);
  const [yPt3, mseTest3] = await evaluateModel(modelOver, xTe, yTe2);

  // Visualisierung (Chart.js)
  function drawChart(canvasId, x, y, pred, title, mse) {
    const ctx = document.createElement("canvas");
    document.getElementById(canvasId).appendChild(ctx);
    new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: "Tatsächliche Werte",
            data: x.map((val, i) => ({ x: val, y: y[i] })),
            backgroundColor: "#2ecc71",
          },
          {
            label: "Vorhersage",
            data: x.map((val, i) => ({ x: val, y: pred[i] })),
            backgroundColor: "#e74c3c",
          },
        ]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: `${title} (MSE: ${mse.toFixed(4)})`
          }
        },
        scales: {
          x: { title: { display: true, text: 'x' } },
          y: { title: { display: true, text: 'y' } }
        }
      }
    });
  }

  drawChart("plot-daten-unverrauscht", xTrain, yTrainClean, yP1, "Train Unverrauscht", mseTrain1);
  drawChart("plot-daten-verrauscht", xTrain, yTrainNoisy, yP2, "Train Verrauscht (Best-Fit)", mseTrain2);
  drawChart("plot-vorhersage-train", xTrain, yTrainNoisy, yP3, "Train Verrauscht (Over-Fit)", mseTrain3);
  drawChart("plot-vorhersage-test", xTest, yTestNoisy, yPt3, "Test Verrauscht (Over-Fit)", mseTest3);
  drawChart("plot-vorhersage-train-clean", xTrain, yTrainClean, yP1, "Train Unverrauscht", mseTrain1);
  drawChart("plot-vorhersage-test-clean", xTest, yTestClean, yPt1, "Test Unverrauscht", mseTest1);
  drawChart("plot-vorhersage-train-best", xTrain, yTrainNoisy, yP2, "Train Verrauscht (Best-Fit)", mseTrain2);
  drawChart("plot-vorhersage-test-best", xTest, yTestNoisy, yPt2, "Test Verrauscht (Best-Fit)", mseTest2);

  // Diskussionstext anzeigen
  const diskussion = `
Modell auf unverrauschten Daten: Train/Test MSE nahezu identisch.
Best-Fit Modell auf verrauschten Daten: gute Generalisierung.
Over-Fit Modell: stark überangepasst, Train-MSE << Test-MSE.
→ Overfitting sichtbar, wenn Trainingsfehler klein, Testfehler hoch.`;

  document.getElementById("diskussionText").textContent = diskussion;

  setTimeout(() => {
    overlay.style.display = "none";
  }, 500);

  document.getElementById("darkModeToggle").addEventListener("click", () => {
    const isDark = document.body.classList.toggle("dark");
    document.getElementById("darkModeToggle").textContent =
      isDark ? "☀️ Light Mode aktivieren" : "🌙 Dark Mode aktivieren";
  });
});
