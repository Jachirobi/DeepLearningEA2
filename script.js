// Starte, sobald die Seite vollständig geladen ist
window.addEventListener("load", async () => {
	// Lade-Overlay anzeigen
	const overlay = document.getElementById("loadingOverlay");
	const loadingText = document.getElementById("loadingText");

	// Anzahl Datenpunkte und Rauschvarianz
	//	initial N = 100;
	//	initial noiseVar = 0.05;
	let N = parseInt(localStorage.getItem("param_N")) || parseInt(document.getElementById("numPoints").value);
	let noiseVar = parseFloat(localStorage.getItem("param_noiseVar")) || parseFloat(document.getElementById("noiseVar").value);

	// sicherstellen, dass auch die UI-Werte aktualisiert sind
	document.getElementById("numPoints").value = N;
	document.getElementById("noiseVar").value = noiseVar;
	document.getElementById("numPointsValue").textContent = N;
	document.getElementById("noiseVarValue").textContent = noiseVar.toFixed(2);
	const seed = 42;

	// Setze Zufallszahlengenerator, falls seedrandom vorhanden
	if (typeof Math.seedrandom === 'function') Math.seedrandom(seed);

	// Ziel-Funktion (5. Ordnung) für Regression
	const f = x => 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;

	// Funktion für gaußsches Rauschen
	function gaussianNoise(mean, stdDev) {
		let u = 0, v = 0;
		while (u === 0) u = Math.random();
		while (v === 0) v = Math.random();
		return stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) + mean;
	}

	// Eingabedaten: gleichmäßig im Bereich [-2, 2]
	const xData = Array.from({ length: N }, () => Math.random() * 4 - 2);
	const yClean = xData.map(f);  // Zielwerte ohne Rauschen
	const yNoisy = yClean.map(y => y + gaussianNoise(0, Math.sqrt(noiseVar)));  // mit Rauschen

	// Indizes für Zufalls-Shuffle → Split in Training und Test
	const indices = Array.from(tf.util.createShuffledIndices(N));
	const splitCount = Math.floor(N / 2);
	const trainIndices = indices.slice(0, splitCount);
	const testIndices = indices.slice(splitCount);

	// Hilfsfunktion zum Indizierten Aufteilen
	const split = (arr, idx) => idx.map(i => arr[i]);
	const xTrain = split(xData, trainIndices);
	const yTrainClean = split(yClean, trainIndices);
	const yTrainNoisy = split(yNoisy, trainIndices);
	const xTest = split(xData, testIndices);
	const yTestClean = split(yClean, testIndices);
	const yTestNoisy = split(yNoisy, testIndices);

	// Konvertiere Eingabe- und Zielwerte in Tensoren
	const toTensor = (x, y) => {
		if (!x || !y || x.length !== y.length || x.length === 0) {
			throw new Error("Ungültige Tensor-Daten: x oder y leer oder unterschiedlich lang.");
		}
		return [tf.tensor2d(x, [x.length, 1]), tf.tensor2d(y, [y.length, 1])];
	};

	// Erzeuge ein FFNN-Modell mit gegebener Architektur
	function createModel(hiddenUnits = [100, 100]) {
		const model = tf.sequential();
		model.add(tf.layers.dense({ units: hiddenUnits[0], activation: 'relu', inputShape: [1] }));
		for (let i = 1; i < hiddenUnits.length; i++) {
			model.add(tf.layers.dense({ units: hiddenUnits[i], activation: 'relu' }));
		}
		model.add(tf.layers.dense({ units: 1 })); // Lineare Ausgabe
		model.compile({
			optimizer: tf.train.adam(0.01),
			loss: 'meanSquaredError'
		});
		return model;
	}

	// Lese Modellarchitektur aus UI (z. B. Slider/Dropdown)
	function getModelArchitectureFromUI() {
		const numLayers = parseInt(document.getElementById("numHiddenLayers")?.value);
		const hiddenUnits = [];
		if (numLayers >= 1) hiddenUnits.push(parseInt(document.getElementById("neuronsLayer1")?.value));
		if (numLayers >= 2) hiddenUnits.push(parseInt(document.getElementById("neuronsLayer2")?.value));
		if (numLayers >= 3) hiddenUnits.push(parseInt(document.getElementById("neuronsLayer3")?.value));
		const valid = hiddenUnits.filter(n => !isNaN(n) && n > 0);
		return valid.length > 0 ? valid : [100, 100];
	}

	// Trainiere Modell auf Tensor-Daten
	const trainModel = async (model, xT, yT, epochs) => {
		await model.fit(xT, yT, { epochs, batchSize: 32, shuffle: true });
	};

	// Evaluiere Modell, berechne MSE und gib Vorhersagen zurück
	const evaluateModel = async (model, xT, yT) => {
		const pred = model.predict(xT);
		const predReshaped = pred.reshape(yT.shape);
		const mseTensor = tf.metrics.meanSquaredError(yT, predReshaped);
		const mse = (await mseTensor.mean().data())[0];  // ⬅️ expliziter Mittelwert
		const yTrueArray = await yT.array();
		const yPredArray = await predReshaped.array();

		console.log("MSE (eval):", mse);
		console.log("  yTrue vs yPred (erste 5):", yTrueArray.slice(0, 5), yPredArray.slice(0, 5));

		console.log("yTrue vs yPred (erste 10 Werte):");
		for (let i = 0; i < 20; i++) {
			console.log(`  ${i}: true=${yTrueArray[i][0].toFixed(4)}  pred=${yPredArray[i][0].toFixed(4)}`);
		}

		return [yPredArray.map(e => e[0]), mse, yTrueArray.map(e => e[0])];
	};

	let lossChart;

	function initLossChart() {
		const container = document.getElementById("loss-chart-container");
		container.innerHTML = ""; // vorherige löschen
		const canvas = document.createElement("canvas");
		container.appendChild(canvas);

		lossChart = new Chart(canvas, {
			type: "line",
			data: {
				labels: [],
				datasets: [
					{
						label: "Train Loss",
						data: [],
						borderColor: "#2ecc71",
						fill: false,
					},
					{
						label: "Test Loss",
						data: [],
						borderColor: "#3498db",
						fill: false,
					},
				],
			},
			options: {
				responsive: true,
				plugins: {
					title: {
						display: true,
						text: "Lernkurve: MSE pro Epoche",
					},
				},
				scales: {
				  x: {
				    type: "linear",  // explizit numerisch
				    title: { display: true, text: "Epoche" }
				  },
					y: { title: { display: true, text: "Loss (MSE)" } },
				},
			},
		});
	}

	let lossChart2;




	function initLossChart2() {
		const container = document.getElementById("loss-chart-container2");
		container.innerHTML = "";
		const canvas = document.createElement("canvas");
		container.appendChild(canvas);

		lossChart2 = new Chart(canvas, {
			type: "line",
			data: {
				labels: [],
				datasets: [
					{
						label: "Train Loss",
						data: [],
						borderColor: "#e67e22",
						fill: false,
					},
					{
						label: "Test Loss",
						data: [],
						borderColor: "#9b59b6",
						fill: false,
					},
				],
			},
			options: {
				responsive: true,
				plugins: {
					title: {
						display: true,
						text: "Lernkurve (Overfit): MSE pro Epoche",
					},
				},
				scales: {
				  x: {
				    type: "linear",  // explizit numerisch
				    title: { display: true, text: "Epoche" }
				  },
					y: { title: { display: true, text: "Loss (MSE)" } },
				},
			},
		});
	}

	// Zeichne Vorhersagediagramm für Trainings- oder Testdaten
	function drawPredictionChart(canvasId, x, y, pred, title, mse, color) {
		const zipped = x.map((val, i) => ({ x: val, y: y[i], pred: pred[i] }));
		zipped.sort((a, b) => a.x - b.x);
		const sx = zipped.map(p => p.x);
		const sy = zipped.map(p => p.y);
		const sp = zipped.map(p => p.pred);
		const ctx = document.createElement("canvas");
		document.getElementById(canvasId).appendChild(ctx);
		new Chart(ctx, {
			type: 'scatter',
			data: {
				datasets: [
					{
						label: "Tatsächliche Werte",
						data: sx.map((val, i) => ({ x: val, y: sy[i] })),
						backgroundColor: color,
						pointStyle: 'circle',
						pointRadius: 4
					},
					{
						label: "Vorhersage",
						data: sx.map((val, i) => ({ x: val, y: sp[i] })),
						backgroundColor: "#e74c3c",
						pointStyle: 'circle',
						pointRadius: 4
					}
				]
			},
			options: {
				plugins: {
					title: {
						display: true,
						text: `${title} (MSE: ${mse.toFixed(6)})`
					},
					legend: { display: true, position: 'bottom' },
					tooltip: {
						callbacks: {
							label: function(context) {
								return `${context.dataset.label}: (x=${context.parsed.x.toFixed(2)}, y=${context.parsed.y.toFixed(2)})`;
							}
						}
					}
				},
				scales: {
					x: { title: { display: true, text: 'x' } },
					y: { title: { display: true, text: 'y' } }
				}
			}
		});
	}

	// Zeichne Diagramm für die Eingabedaten (Train/Test)
	function drawDatasetChart(canvasId, xTrain, yTrain, xTest, yTest, title) {
		const ctx = document.createElement("canvas");
		document.getElementById(canvasId).appendChild(ctx);
		new Chart(ctx, {
			type: 'scatter',
			data: {
				datasets: [
					{
						label: "Trainingsdaten",
						data: xTrain.map((val, i) => ({ x: val, y: yTrain[i] })),
						backgroundColor: "#2ecc71",
						pointStyle: 'circle',
						pointRadius: 4
					},
					{
						label: "Testdaten",
						data: xTest.map((val, i) => ({ x: val, y: yTest[i] })),
						backgroundColor: "#3498db",
						pointStyle: 'circle',
						pointRadius: 4
					}
				]
			},
			options: {
				plugins: {
					title: { display: true, text: title },
					legend: { display: true, position: 'bottom' },
					tooltip: {
						enabled: true,
						callbacks: {
							label: function(context) {
								return `${context.dataset.label}: (x=${context.parsed.x.toFixed(2)}, y=${context.parsed.y.toFixed(2)})`;
							}
						}
					}
				},
				scales: {
					x: { title: { display: true, text: 'x' } },
					y: { title: { display: true, text: 'y' } }
				}
			}
		});
	}

	// Initialisiere zwei Modelle (Clean, Overfit)
	const architecture = getModelArchitectureFromUI();  // einmal auslesen
	const modelClean = createModel(architecture);
	const modelOver = createModel(architecture);

	loadingText.textContent = "📊 Trainiere Modelle...";

	// Tensor-Konvertierung für alle Daten
	const [xT1, yT1] = toTensor(xTrain, yTrainClean);
	const [xT2, yT2] = toTensor(xTrain, yTrainNoisy);
	const [xTe, yTe1] = toTensor(xTest, yTestClean);
	const [_, yTe2] = toTensor(xTest, yTestNoisy);

	// Modelle trainieren
	await trainModel(modelClean, xT1, yT1, 100);
	await trainModel(modelOver, xT2, yT2, 300);

	// Modelle evaluieren
	const [yP1, mseTrain1] = await evaluateModel(modelClean, xT1, yT1);
	const [yPt1, mseTest1] = await evaluateModel(modelClean, xTe, yTe1);
	const [yP3, mseTrain3] = await evaluateModel(modelOver, xT2, yT2);
	const [yPt3, mseTest3] = await evaluateModel(modelOver, xTe, yTe2);

	// Funktion zum Trainieren und Zeichnen des Best-Fit-Modells
	async function trainAndDrawBestFit(epochs) {
		const hiddenUnits = getModelArchitectureFromUI();
		const modelBest = createModel(hiddenUnits);
		//    await trainModel(modelBest, xT2, yT2, epochs);

		if (lossChart) {
		  lossChart.destroy(); // vorherige Instanz entfernen
		}
		initLossChart(); // neuen Chart aufsetzen
		
		// WICHTIG: Chart-Daten vollständig zurücksetzen
		lossChart.data.labels = [];
		lossChart.data.datasets.forEach(ds => ds.data = []);
		lossChart.update();

		for (let epoch = 1; epoch <= epochs; epoch++) {
		    const history = await modelBest.fit(xT2, yT2, { epochs: 1, shuffle: true });

		    // Verwende evaluateModel für konsistente MSE-Werte
		    const [_, mseTrain] = await evaluateModel(modelBest, xT2, yT2);
		    const [__, mseTest] = await evaluateModel(modelBest, xTe, yTe2);

		    lossChart.data.labels.push(Number(epoch));
		    lossChart.data.datasets[0].data.push(mseTrain);
		    lossChart.data.datasets[1].data.push(mseTest);
		    lossChart.update();
		}

		// ALLES aus Tensoren holen – synchron!
		const [yP2, mseTrain2] = await evaluateModel(modelBest, xT2, yT2);
		const [yPt2, mseTest2] = await evaluateModel(modelBest, xTe, yTe2);

		const xTrainSorted = (await xT2.array()).map(e => e[0]);
		const yTrainSorted = (await yT2.array()).map(e => e[0]);
		const xTestSorted = (await xTe.array()).map(e => e[0]);

		console.log(`📈 [Best-Fit] Epochen: ${epochs}`);
		console.log(`   ↳ MSE Train: ${mseTrain2.toFixed(6)}`);
		console.log(`   ↳ MSE Test:  ${mseTest2.toFixed(6)}`);
		console.log("   ↳ Architektur:", hiddenUnits);

		document.getElementById("plot-vorhersage-train-best").innerHTML = "";
		document.getElementById("plot-vorhersage-test-best").innerHTML = "";

		drawPredictionChart(
			"plot-vorhersage-train-best",
			xTrainSorted,
			yTrainSorted,
			yP2,
			`Train Verrauscht (Best-Fit)`,
			mseTrain2,
			"#2ecc71"
		);

		drawPredictionChart(
			"plot-vorhersage-test-best",
			xTestSorted,
			yTestNoisy,
			yPt2,
			`Test Verrauscht (Best-Fit)`,
			mseTest2,
			"#3498db"
		);
	}

	//Training für Overfit - Modell mit Lernkurve
	async function trainAndDrawOverfit(epochs) {
		const hiddenUnits = getModelArchitectureFromUI();
		const modelOverfit = createModel(hiddenUnits);

		if (lossChart2) {
		  lossChart2.destroy(); // vorherige Instanz entfernen
		}
		initLossChart2(); // neuen Chart aufsetzen

		// NEU: Chart-Daten zurücksetzen
		lossChart2.data.labels = [];
		lossChart2.data.datasets.forEach(ds => ds.data = []);
		lossChart2.update();

		for (let epoch = 1; epoch <= epochs; epoch++) {
		    const history = await modelOverfit.fit(xT2, yT2, { epochs: 1, shuffle: true });

		    // Verwende evaluateModel wie im Best-Fit
		    const [_, mseTrain] = await evaluateModel(modelOverfit, xT2, yT2);
		    const [__, mseTest] = await evaluateModel(modelOverfit, xTe, yTe2);

		    lossChart2.data.labels.push(Number(epoch));
		    lossChart2.data.datasets[0].data.push(mseTrain);
		    lossChart2.data.datasets[1].data.push(mseTest);
		    lossChart2.update();
		}

		// ALLES aus Tensoren holen – synchron!
		const [yP4, mseTrain4] = await evaluateModel(modelOverfit, xT2, yT2);
		const [yPt4, mseTest4] = await evaluateModel(modelOverfit, xTe, yTe2);
		const xTrainSorted = (await xT2.array()).map(e => e[0]);
		const yTrainSorted = (await yT2.array()).map(e => e[0]);
		const xTestSorted = (await xTe.array()).map(e => e[0]);

		console.log(`📈 [Overfit] Epochen: ${epochs}`);
		console.log(`   ↳ MSE Train: ${mseTrain4.toFixed(6)}`);
		console.log(`   ↳ MSE Test:  ${mseTest4.toFixed(6)}`);
		console.log("   ↳ Architektur:", hiddenUnits);

		document.getElementById("plot-vorhersage-train").innerHTML = "";
		document.getElementById("plot-vorhersage-test").innerHTML = "";

		drawPredictionChart(
			"plot-vorhersage-train",
			xTrainSorted,
			yTrainSorted,
			yP4,
			`Train Verrauscht (Overfit)`,
			mseTrain4,
			"#2ecc71"
		);

		drawPredictionChart(
			"plot-vorhersage-test",
			xTestSorted,
			yTestNoisy,
			yPt4,
			`Test Verrauscht (Overfit)`,
			mseTest4,
			"#3498db"
		);
	}

	// Initiales Training (Best-Fit)
	const initialEpochs = 50;
	setTimeout(() => trainAndDrawBestFit(initialEpochs), 0);  // Sicherstellen, dass DOM bereit ist

	// Initiales Training (Overfit)
	const initialOverfitEpochs = 300;
	setTimeout(() => trainAndDrawOverfit(initialOverfitEpochs), 0);

	// Daten- und Vorhersage-Diagramme zeichnen
	drawDatasetChart("plot-daten-unverrauscht", xTrain, yTrainClean, xTest, yTestClean, "Unverrauschte Daten: Train (grün) / Test (blau)");
	drawDatasetChart("plot-daten-verrauscht", xTrain, yTrainNoisy, xTest, yTestNoisy, "Verrauschte Daten: Train (grün) / Test (blau)");

	drawPredictionChart("plot-vorhersage-train-clean", xTrain, yTrainClean, yP1, "Train Unverrauscht", mseTrain1, "#2ecc71");
	drawPredictionChart("plot-vorhersage-test-clean", xTest, yTestClean, yPt1, "Test Unverrauscht", mseTest1, "#3498db");

	drawPredictionChart("plot-vorhersage-train", xTrain, yTrainNoisy, yP3, "Train Verrauscht (Over-Fit)", mseTrain3, "#2ecc71");
	drawPredictionChart("plot-vorhersage-test", xTest, yTestNoisy, yPt3, "Test Verrauscht (Over-Fit)", mseTest3, "#3498db");

	// Epochen-Slider initialisieren
	const epochSlider = document.getElementById("epochSlider");
	const epochValue = document.getElementById("epochValue");
	let debounceTimer;
	epochSlider.addEventListener("input", (e) => {
	  const val = parseInt(e.target.value);
	  epochValue.textContent = val;

	  clearTimeout(debounceTimer);
	  debounceTimer = setTimeout(() => {
	    trainAndDrawBestFit(val);
	  }, 500); // erst nach 500ms Stillstand trainieren
	});

	async function reinitializeAndTrainAllModels() {
		const architecture = getModelArchitectureFromUI();

		// Alte Plots leeren
		document.getElementById("plot-vorhersage-train-clean").innerHTML = "";
		document.getElementById("plot-vorhersage-test-clean").innerHTML = "";
		document.getElementById("plot-vorhersage-train").innerHTML = "";
		document.getElementById("plot-vorhersage-test").innerHTML = "";

		// Modelle neu erstellen
		const modelClean = createModel(architecture);
		const modelOver = createModel(architecture);

		loadingText.textContent = "📊 Trainiere Modelle...";

		// Neu trainieren
		await trainModel(modelClean, xT1, yT1, 100);
		await trainModel(modelOver, xT2, yT2, 300);

		// Neu evaluieren
		const [yP1, mseTrain1] = await evaluateModel(modelClean, xT1, yT1);
		const [yPt1, mseTest1] = await evaluateModel(modelClean, xTe, yTe1);
		const [yP3, mseTrain3] = await evaluateModel(modelOver, xT2, yT2);
		const [yPt3, mseTest3] = await evaluateModel(modelOver, xTe, yTe2);

		// Plots neu zeichnen
		drawPredictionChart("plot-vorhersage-train-clean", xTrain, yTrainClean, yP1, "Train Unverrauscht", mseTrain1, "#2ecc71");
		drawPredictionChart("plot-vorhersage-test-clean", xTest, yTestClean, yPt1, "Test Unverrauscht", mseTest1, "#3498db");
		drawPredictionChart("plot-vorhersage-train", xTrain, yTrainNoisy, yP3, "Train Verrauscht (Over-Fit)", mseTrain3, "#2ecc71");
		drawPredictionChart("plot-vorhersage-test", xTest, yTestNoisy, yPt3, "Test Verrauscht (Over-Fit)", mseTest3, "#3498db");

		loadingText.textContent = "✅ Modelle aktualisiert";
		setTimeout(() => overlay.style.display = "none", 500);
	}

	document.getElementById("applyArchitecture").addEventListener("click", async () => {
		overlay.style.display = "flex";
		await reinitializeAndTrainAllModels();
	});


	// Diskussions-Text anzeigen
	const diskussion = `
Modell auf unverrauschten Daten: Train/Test MSE nahezu identisch.
Best-Fit Modell auf verrauschten Daten: gute Generalisierung.
Over-Fit Modell: stark überangepasst, Train-MSE << Test-MSE.
→ Overfitting sichtbar, wenn Trainingsfehler klein, Testfehler hoch.`;

	document.getElementById("numPoints").addEventListener("input", e => {
		document.getElementById("numPointsValue").textContent = e.target.value;
	});

	document.getElementById("noiseVar").addEventListener("input", e => {
		document.getElementById("noiseVarValue").textContent = e.target.value;
	});
	
	document.getElementById("applyDataParams").addEventListener("click", async () => {
		N = parseInt(document.getElementById("numPoints").value);
		noiseVar = parseFloat(document.getElementById("noiseVar").value);

		// SPEICHERN in localStorage
		localStorage.setItem("param_N", N);
		localStorage.setItem("param_noiseVar", noiseVar);

		overlay.style.display = "flex";
		loadingText.textContent = "📊 Generiere neue Daten...";

		location.reload();
	});

	document.getElementById("diskussionText").textContent = diskussion;

	// Ladeanzeige ausblenden
	setTimeout(() => {
		overlay.style.display = "none";
	}, 500);

	// Dark Mode Toggle
	document.getElementById("darkModeToggle").addEventListener("click", () => {
		const isDark = document.body.classList.toggle("dark");
		document.getElementById("darkModeToggle").textContent =
			isDark ? "☀️ Light Mode aktivieren" : "🌙 Dark Mode aktivieren";
	});

	let debounceTimer2;
	document.getElementById("epochSlider2").addEventListener("input", async (e) => {
		const val = parseInt(e.target.value);
		document.getElementById("epochValue2").textContent = val;
		
		clearTimeout(debounceTimer2);
		debounceTimer2 = setTimeout(() => {
		  trainAndDrawOverfit(val);
		}, 500); // erst nach 500ms Stillstand trainieren

	});
	
	// Ein-/Ausklapp-Logik
	document.querySelectorAll(".collapsible-section .toggle-button").forEach(btn => {
	  btn.addEventListener("click", () => {
	    const section = btn.closest(".collapsible-section");
	    section.classList.toggle("collapsed");
	    btn.textContent = section.classList.contains("collapsed") ? "⬆️ Ausklappen" : "⬇️ Einklappen";
	  });
	});
	

});
