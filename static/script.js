const form = document.getElementById("predictionForm");
const resultBox = document.getElementById("resultBox");
const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  resizeDelay: 100,
  plugins: {
    legend: {
      display: false,
    },
  },
  scales: {
    y: {
      beginAtZero: true,
      ticks: {
        precision: 0,
      },
    },
  },
};

function renderBarChart(canvasId, labels, values, label) {
  const canvas = document.getElementById(canvasId);
  const oldChart = Chart.getChart(canvas);

  if (oldChart) {
    oldChart.destroy();
  }

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: label,
          data: values,
          borderWidth: 1,
          borderRadius: 8,
        },
      ],
    },
    options: chartOptions,
  });
}

async function loadDashboard() {
  try {
    const response = await fetch("/dashboard-data");
    const data = await response.json();

    if (!response.ok) {
      return;
    }

    document.getElementById("totalRecords").textContent = data.insights.total_records;
    document.getElementById("averageMarks").textContent = `${data.insights.average_marks}/100`;
    document.getElementById("averageAttendance").textContent = `${data.insights.average_attendance}%`;

    renderBarChart(
      "attendanceChart",
      data.attendance_distribution.map((item) => item.label),
      data.attendance_distribution.map((item) => item.count),
      "Students"
    );

    renderBarChart(
      "marksChart",
      data.marks_distribution.map((item) => item.label),
      data.marks_distribution.map((item) => item.count),
      "Students"
    );

    renderBarChart(
      "performanceChart",
      data.performance_categories.map((item) => item.grade),
      data.performance_categories.map((item) => item.count),
      "Students"
    );
  } catch (error) {
    console.log("Dashboard data could not be loaded.");
  }
}

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  resultBox.innerHTML = "Processing prediction...";
  resultBox.className = "result-box loading";

  const formData = new FormData(form);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      resultBox.innerHTML = data.error || "Something went wrong.";
      resultBox.className = "result-box error";
      return;
    }

    resultBox.innerHTML = `
      <h3>${data.message}</h3>
      <p><strong>Predicted Score:</strong> ${data.predicted_score}/100</p>
      <p><strong>Model Used:</strong> ${data.model_used}</p>
    `;
    resultBox.className = "result-box success";
  } catch (error) {
    resultBox.innerHTML = "Unable to connect to the server. Please run Flask app first.";
    resultBox.className = "result-box error";
  }
});

loadDashboard();
