import os
import uuid
import time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from config import Config
from model.predictor import DeepfakePredictor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model/weights', exist_ok=True)

# Initialize predictor
print("Initializing Deepfake Predictor...")
predictor = DeepfakePredictor(
    model_path=app.config.get('MODEL_PATH'),
    device=app.config.get('DEVICE', 'cuda'),
    image_size=app.config.get('IMAGE_SIZE', 224)
)


def get_file_type(filename):
    """Determine if file is image or video"""
    if '.' not in filename:
        return None
        
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext in Config.ALLOWED_IMAGE_EXTENSIONS:
        return 'image'
    elif ext in Config.ALLOWED_VIDEO_EXTENSIONS:
        return 'video'
    return None


@app.route('/')
def index():
    """Home page"""
    model_info = predictor.get_model_info()
    return render_template('index.html', model_info=model_info)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded file for deepfakes"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Determine file type
    file_type = get_file_type(file.filename)
    
    if file_type is None:
        return jsonify({
            'error': 'Invalid file type. Allowed: images (png, jpg, jpeg, webp) and videos (mp4, avi, mov, mkv)'
        }), 400
    
    # Generate unique filename
    original_filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())[:8]
    ext = original_filename.rsplit('.', 1)[1].lower()
    filename = f"{unique_id}_{int(time.time())}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Save file
        file.save(filepath)
        
        # Analyze
        start_time = time.time()
        
        if file_type == 'image':
            result = predictor.predict_image(filepath)
        else:
            result = predictor.predict_video(filepath)
        
        processing_time = round(time.time() - start_time, 2)
        
        # Add metadata
        result['filename'] = original_filename
        result['file_type'] = file_type
        result['processing_time'] = processing_time
        result['file_url'] = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up file on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
def model_info():
    """Get model information"""
    return jsonify(predictor.get_model_info())


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'gpu_available': predictor.get_model_info()['gpu_available'],
        'model_loaded': predictor.get_model_info()['model_loaded']
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large. Maximum size is 100MB'}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("        DEEPFAKE ANALYZER - Starting Server")
    print("=" * 60)
    info = predictor.get_model_info()
    print(f"  Device      : {info['device']}")
    print(f"  GPU         : {info['gpu_name'] or 'Not available'}")
    print(f"  Model Loaded: {info['model_loaded']}")
    print(f"  Parameters  : {info['total_parameters']:,}")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )