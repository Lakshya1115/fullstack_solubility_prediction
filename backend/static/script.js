// Event listener for form submission
document.getElementById("predict-form").addEventListener("submit", function (event) {
    event.preventDefault();  // Prevent the default form submission

    // Get the feature values from the form inputs
    const feature1 = parseFloat(document.getElementById('feature1').value);
    const feature2 = parseFloat(document.getElementById('feature2').value);
    const feature3 = parseFloat(document.getElementById('feature3').value);
    const feature4 = parseFloat(document.getElementById('feature4').value);

    // Prepare the data to send in the request
    const requestData = {
        features: [feature1, feature2, feature3, feature4]
    };

    // Send a POST request to the Flask backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            // Display the predictions returned by the backend
            document.getElementById('lr-prediction').innerText = `Linear Regression Prediction: ${data.linear_regression}`;
            document.getElementById('rf-prediction').innerText = `Random Forest Prediction: ${data.random_forest}`;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred: ' + error.message);
        });
});