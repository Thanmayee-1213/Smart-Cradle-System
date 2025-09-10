from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import pickle
import librosa
import numpy as np
import pandas as pd
import io
import base64
import wave
import tempfile
import os
from datetime import datetime, timedelta
import threading
import time
import json
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

app = Flask(_name_)
app.config['SECRET_KEY'] = 'baby_cry_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class DatabaseManager:
    def _init_(self):
        self.db_path = 'baby_cry_data.db'
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                feedback TEXT DEFAULT NULL,
                correct_class TEXT DEFAULT NULL,
                is_correct BOOLEAN DEFAULT NULL
            )
        ''')
        
        # Training data table for retraining
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                audio_features TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                correct_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                used_for_training BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, prediction_data):
        """Save prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy types before JSON serialization
        probabilities = convert_numpy_types(prediction_data['probabilities'])
        
        cursor.execute('''
            INSERT INTO predictions (timestamp, predicted_class, confidence, probabilities)
            VALUES (?, ?, ?, ?)
        ''', (
            prediction_data['timestamp'],
            prediction_data['predicted_class'],
            float(prediction_data['confidence']),  # Ensure it's a Python float
            json.dumps(probabilities)
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def update_prediction_feedback(self, prediction_id, feedback, correct_class):
        """Update prediction with user feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        is_correct = feedback == 'correct'
        
        cursor.execute('''
            UPDATE predictions 
            SET feedback = ?, correct_class = ?, is_correct = ?
            WHERE id = ?
        ''', (feedback, correct_class, is_correct, prediction_id))
        
        conn.commit()
        conn.close()
    
    def save_training_feedback(self, audio_features, predicted_class, correct_class, confidence):
        """Save feedback data for retraining"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy types in features
        features_converted = convert_numpy_types(audio_features)
        
        cursor.execute('''
            INSERT INTO training_feedback 
            (timestamp, audio_features, predicted_class, correct_class, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(features_converted),
            predicted_class,
            correct_class,
            float(confidence)  # Ensure it's a Python float
        ))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self, days=30):
        """Get prediction statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Total predictions
        cursor.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE timestamp >= ?
        ''', (start_date.isoformat(),))
        total_predictions = cursor.fetchone()[0]
        
        # Predictions by class
        cursor.execute('''
            SELECT predicted_class, COUNT(*) as count
            FROM predictions 
            WHERE timestamp >= ?
            GROUP BY predicted_class
            ORDER BY count DESC
        ''', (start_date.isoformat(),))
        predictions_by_class = dict(cursor.fetchall())
        
        # Accuracy statistics (where feedback is available)
        cursor.execute('''
            SELECT 
                COUNT(*) as total_with_feedback,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_predictions
            FROM predictions 
            WHERE timestamp >= ? AND feedback IS NOT NULL
        ''', (start_date.isoformat(),))
        accuracy_data = cursor.fetchone()
        
        accuracy = 0
        if accuracy_data[0] > 0:
            accuracy = (accuracy_data[1] / accuracy_data[0]) * 100
        
        # Daily prediction counts
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count
            FROM predictions 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (start_date.isoformat(),))
        daily_counts = cursor.fetchall()
        
        # Confidence distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN confidence >= 0.9 THEN 'Very High (90-100%)'
                    WHEN confidence >= 0.7 THEN 'High (70-89%)'
                    WHEN confidence >= 0.5 THEN 'Medium (50-69%)'
                    ELSE 'Low (<50%)'
                END as confidence_range,
                COUNT(*) as count
            FROM predictions 
            WHERE timestamp >= ?
            GROUP BY confidence_range
        ''', (start_date.isoformat(),))
        confidence_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'predictions_by_class': predictions_by_class,
            'accuracy': round(accuracy, 2),
            'daily_counts': daily_counts,
            'confidence_distribution': confidence_distribution,
            'feedback_count': accuracy_data[0]
        }

class BabyCryPredictor:
    def _init_(self):
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.db_manager = DatabaseManager()
        self.load_model()
        self.low_confidence_threshold = 0.6
        self.retrain_threshold = 10  # Retrain after 10 low-confidence feedbacks
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('baby_cry_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
                self.feature_columns = model_data['feature_columns']
            logger.info("Model loaded successfully!")
        except FileNotFoundError:
            logger.error("Model file not found! Please run main.py first to train the model.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def extract_features(self, audio_data, sr=22050):
        """Extract features from audio data"""
        try:
            features = {}
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
                features[f'mfcc_std_{i}'] = float(np.std(mfccs[i]))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # RMS Energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features['rms'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            for i in range(12):
                features[f'chroma_{i}'] = float(np.mean(chroma[i]))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            features['tempo'] = float(tempo)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio_data, sr=22050):
        """Predict baby cry type from audio data"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        features = self.extract_features(audio_data, sr)
        if features is None:
            return {"error": "Could not extract features"}
        
        # Convert to DataFrame with correct column order
        feature_df = pd.DataFrame([features])
        feature_df = feature_df.reindex(columns=self.feature_columns, fill_value=0)
        
        # Make prediction
        prediction_proba = self.model.predict_proba(feature_df)[0]
        prediction = self.model.predict(feature_df)[0]
        
        # Decode label
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities for all classes - convert to native Python types
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = float(prediction_proba[i])
        
        result = {
            'predicted_class': str(predicted_class),  # Ensure it's a string
            'confidence': float(np.max(prediction_proba)),  # Convert to Python float
            'probabilities': class_probabilities,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': features  # Features are already converted to Python types
        }
        
        # Save prediction to database
        prediction_id = self.db_manager.save_prediction(result)
        result['prediction_id'] = int(prediction_id)  # Ensure it's a Python int
        
        return result
    
    def process_feedback(self, prediction_id, feedback_type, correct_class=None):
        """Process user feedback and determine if retraining is needed"""
        try:
            # Update prediction with feedback
            self.db_manager.update_prediction_feedback(prediction_id, feedback_type, correct_class)
            
            # If feedback indicates incorrect prediction, save for retraining
            if feedback_type == 'incorrect' and correct_class:
                # Get original prediction data
                conn = sqlite3.connect(self.db_manager.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT predicted_class, confidence, probabilities 
                    FROM predictions WHERE id = ?
                ''', (prediction_id,))
                pred_data = cursor.fetchone()
                conn.close()
                
                if pred_data:
                    # Save for retraining if confidence was low
                    if pred_data[1] < self.low_confidence_threshold:
                        # Note: In a real implementation, you'd need to store the audio features
                        # For now, we'll just track the feedback
                        logger.info(f"Low confidence incorrect prediction saved for retraining")
                        
                        # Check if we should retrain
                        self.check_retrain_condition()
            
            return {"status": "success", "message": "Feedback recorded"}
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_retrain_condition(self):
        """Check if model should be retrained based on feedback"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        # Count low confidence incorrect predictions in last 24 hours
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE timestamp >= ? AND is_correct = 0 AND confidence < ?
        ''', (yesterday, self.low_confidence_threshold))
        
        low_conf_incorrect = cursor.fetchone()[0]
        conn.close()
        
        if low_conf_incorrect >= self.retrain_threshold:
            logger.info(f"Retrain condition met: {low_conf_incorrect} low confidence incorrect predictions")
            # In a production system, you would trigger retraining here
            # For now, we'll just log it
            return True
        
        return False

# Initialize predictor and database
predictor = BabyCryPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_audio():
    """Handle audio file prediction"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'})
        
        audio_file = request.files['audio']
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_filename = tmp_file.name
            audio_file.save(tmp_filename)

        audio_data, sr = librosa.load(tmp_filename, duration=3, sr=22050)
        os.unlink(tmp_filename)

        result = predictor.predict(audio_data, sr)
        
        # Ensure all values are JSON serializable
        result = convert_numpy_types(result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_audio: {e}")
        return jsonify({'error': str(e)})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Handle user feedback"""
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        feedback_type = data.get('feedback_type')  # 'correct' or 'incorrect'
        correct_class = data.get('correct_class')  # If incorrect, what's the correct class
        
        result = predictor.process_feedback(prediction_id, feedback_type, correct_class)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in submit_feedback: {e}")
        return jsonify({'error': str(e)})

@app.route('/statistics')
def get_statistics():
    """Get prediction statistics"""
    try:
        days = request.args.get('days', 30, type=int)
        stats = predictor.db_manager.get_statistics(days)
        
        # Ensure all values are JSON serializable
        stats = convert_numpy_types(stats)
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in get_statistics: {e}")
        return jsonify({'error': str(e)})

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle real-time audio data from client"""
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(data['audio'])
        
        # Convert to audio array
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_filename = tmp_file.name
            tmp_file.write(audio_bytes)
            tmp_file.flush()

        audio_data, sr = librosa.load(tmp_filename, duration=3, sr=22050)
        os.unlink(tmp_filename)

        # Make prediction
        result = predictor.predict(audio_data, sr)
        
        # Ensure all values are JSON serializable
        result = convert_numpy_types(result)
        
        # Emit result back to client
        emit('prediction_result', result)
        
    except Exception as e:
        logger.error(f"Error in handle_audio_data: {e}")
        emit('prediction_result', {'error': str(e)})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if _name_ == '_main_':
    print("Starting Baby Cry Classification Server...")
    print("Make sure you have trained the model using main.py first!")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)