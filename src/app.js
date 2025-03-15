document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('diagnosis-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Collect form data
            const formData = {
                Pregnancies: document.getElementById('Pregnancies').value,
                Glucose: document.getElementById('Glucose').value,
                BloodPressure: document.getElementById('BloodPressure').value,
                SkinThickness: document.getElementById('SkinThickness').value,
                Insulin: document.getElementById('Insulin').value,
                BMI: document.getElementById('BMI').value,
                DiabetesPedigreeFunction: document.getElementById('DiabetesPedigreeFunction').value,
                Age: document.getElementById('Age').value
            };

            try {
                // Send data to Flask backend
                const response = await axios.post('http://localhost:5000/api/predict', formData);
                const { probability, outcome } = response.data;

                // Store result in sessionStorage and redirect
                sessionStorage.setItem('probability', probability);
                sessionStorage.setItem('outcome', outcome);
                window.location.href = '/result';
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while predicting.');
            }
        });
    }

    // Populate result page
    const probabilitySpan = document.getElementById('probability');
    const outcomeSpan = document.getElementById('outcome');
    if (probabilitySpan && outcomeSpan) {
        probabilitySpan.textContent = sessionStorage.getItem('probability');
        outcomeSpan.textContent = sessionStorage.getItem('outcome');
    }
});