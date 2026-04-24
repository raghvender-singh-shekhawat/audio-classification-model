// ===============================
// LOAD ONNX MODEL
// ===============================
let session = null;

export async function loadModel() {
  try {
    session = await ort.InferenceSession.create("model.onnx");
    console.log("✅ ONNX model loaded");
  } catch (err) {
    console.error("❌ Model load error:", err);
  }
}

// ===============================
// RUN MODEL
// ===============================
export async function runModel(featuresArray) {
  if (!session) {
    console.error("Model not loaded yet!");
    return null;
  }

  if (featuresArray.length !== 80) {
    console.error("Expected 80 features, got:", featuresArray.length);
    return null;
  }

  try {
    const inputTensor = new ort.Tensor(
      "float32",
      Float32Array.from(featuresArray),
      [1, 80]
    );

    const results = await session.run({ input: inputTensor });

    const output = results.output.data;

    return {
      real: output[0],
      fake: output[1]
    };

  } catch (err) {
    console.error("❌ Inference error:", err);
    return null;
  }
}