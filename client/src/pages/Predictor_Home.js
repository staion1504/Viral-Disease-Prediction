import React, { useState } from 'react';

function Predictor_Home() {
    const [sequence, setSequence] = useState('');
    const [prediction, setPrediction] = useState('');

    const handleSubmit = async (event) => {
        event.preventDefault(); // Prevent default form submission
      
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ sequence: sequence })
            });

            const data = await response.json();
            if(data.prediction==1){
              setPrediction("HIV");
            }
            else if(data.prediction==2){
              setPrediction("ROTA");
            }
            else if(data.prediction==3){
              setPrediction("HEPATITIS B");
            }
            else {
              setPrediction("EBOLA");
            }
            
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return (
        <div style={{ width: '50%', margin: 'auto', marginTop: '50px' }}>
            <h2 style={{ fontFamily: 'Arial, sans-serif', textAlign: 'center' }}>Viral Disease Predictor</h2>
            <form style={{ marginTop: '20px' }} onSubmit={handleSubmit}>
                <textarea
                    style={{ width: '100%', height: '150px', padding: '10px', fontSize: '16px' }}
                    placeholder="Enter DNA sequence..."
                    value={sequence}
                    onChange={(e) => setSequence(e.target.value)}
                ></textarea>
                <button type="submit" style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}>Predict</button>
            </form>
            {prediction && (
                <div style={{ color: 'green', fontWeight: 'bold', marginTop: '20px', fontSize: '18px', textAlign: 'center' }}>
                    <strong>Predicted Disease:</strong> {prediction}
                </div>
            )}
        </div>
    );
}

export default Predictor_Home;
