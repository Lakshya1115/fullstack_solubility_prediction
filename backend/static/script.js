document.getElementById("predictForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    
    let inputData = document.getElementById("features").value.split(",").map(Number);
    
    let response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: inputData })
    });

    let result = await response.json();
    document.getElementById("result").innerText = `Predictions:
    \nLinear Regression: ${result["Linear Regression"]}
    \nRandom Forest: ${result["Random Forest"]}`;
});
