document.addEventListener('DOMContentLoaded', function () {
    // Handle form submission
    document.getElementById('analyzerForm').addEventListener('submit', function(e) {
        e.preventDefault();

        // Get user input values
        const userReturn = parseFloat(document.getElementById('return').value);
        const userRisk = parseInt(document.getElementById('risk').value);

        // Send the data to the Flask backend for analysis
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ return: userReturn, risk: userRisk })
        })
        .then(response => response.json())
        .then(data => {
            console.log("Data received from backend:", data);
            if (data.stocks && data.stocks.length > 0) {
                displayResults(data);
            } else {
                console.error("No stocks received or incorrect data format.");
            }
        })
        .catch(error => console.error('Error:', error));
    });

    // Function to display the results in a table and chart
    function displayResults(data) {
        const recommendationTableBody = document.querySelector('#recommendationTable tbody');
        recommendationTableBody.innerHTML = '';  // Clear any previous rows

        // Loop through the recommended stocks and append them to the table
        data.stocks.forEach(stock => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${stock.Ticker}</td>
                <td>${stock['1-Year Return']}</td>
                <td>${stock['Risk Score']}</td>
                <td>${stock['Suitability Score'].toFixed(2)}</td>
            `;
            recommendationTableBody.appendChild(row);
        });

        // Display the results section
        document.getElementById('results').style.display = 'block';

        // Prepare the chart data
        const ctx = document.getElementById('scoreChart').getContext('2d');
        const chartData = {
            labels: data.stocks.map(stock => stock.Ticker),
            datasets: [{
                label: 'Suitability Score',
                data: data.stocks.map(stock => stock['Suitability Score']),
                borderColor: '#2563eb',
                fill: false
            }]
        };

        // If a chart exists, destroy it before creating a new one
        if (window.scoreChart) {
            window.scoreChart.destroy();
        }

        // Create a new chart
        window.scoreChart = new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 10
                    }
                }
            }
        });
    }
});
