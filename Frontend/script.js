const API_URL = "http://127.0.0.1:8000/predict";

const fields = [
  "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
];

function showResult(cls, html) {
  const box = document.getElementById("result");
  box.className = cls;
  box.innerHTML = html;
  box.style.display = "block";
}

async function checkRisk() {
  const btn = document.getElementById("btn");
  const data = {};

  // Read the values from the form (empty = 0 = "missing", the model handles it)
  for (const f of fields) {
    const v = document.getElementById(f).value.trim();
    data[f] = v === "" ? 0 : Number(v);
  }

  if (data.Age <= 0) {
    showResult("error", "<p class='msg'>Please enter your age.</p>");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Checking...";

  try {
    // Send the values to the Python API
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error("Server returned " + res.status);
    const out = await res.json();

    const pct = Math.round(out.diabetes_risk * 100);
    let cls, title, advice;

    if (pct < 30) {
      cls = "low";
      title = "Lower risk";
      advice = "Your values suggest a lower risk. Keep up healthy habits and get regular checkups.";
    } else if (pct < 50) {
      cls = "mid";
      title = "Moderate risk";
      advice = "Your values show a moderate risk. Consider talking to a doctor about a blood sugar test.";
    } else {
      cls = "high";
      title = "Higher risk";
      advice = "Your values suggest a higher risk. Please consult a doctor for a proper test.";
    }

    showResult(cls,
      "<p class='big'>" + pct + "%</p>" +
      "<p class='msg'><b>" + title + "</b>. " + advice + "</p>" +
      "<div class='bar'><div id='fill'></div></div>");
    setTimeout(() => { document.getElementById("fill").style.width = pct + "%"; }, 50);

  } catch (err) {
    showResult("error",
      "<p class='msg'><b>Could not reach the model.</b> Make sure the Python backend is running:<br>" +
      "<code>python -m uvicorn main:app --reload</code><br>inside the Backend folder.</p>");
  } finally {
    btn.disabled = false;
    btn.textContent = "Check my risk";
  }
}