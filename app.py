from flask import Flask, render_template, request, jsonify, send_file, Response
from thesis_evaluation import PaperAnnotator, Config, cleanup_old_files, ProcessStatus
import os
import uuid
import logging
import re
import shutil
import json
import time
from utils.progress import progress_queue
import zipfile
import socket

app = Flask(__name__, static_folder='temp')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_papers():
    """
    Process uploaded papers through the annotation pipeline.
    
    Handles file upload, model selection, role assignment, 
    and initiates paper processing.
    
    Returns:
        JSON response with processing status and session ID
    """
    session_input_dir = None
    try:
        # Initialize required directories
        Config.init_dirs()
        
        # Generate unique session identifier
        session_id = str(uuid.uuid4())
        session_input_dir = os.path.join(Config.INPUT_DIR, session_id)
        session_output_dir = os.path.join(Config.OUTPUT_DIR, session_id)
        
        # Clear previous progress queue
        while not progress_queue.empty():
            progress_queue.get()
        
        logging.info(f"Creating session directories for ID: {session_id}")
        os.makedirs(session_input_dir)
        os.makedirs(session_output_dir)
        
        # Validate file upload
        if 'files' not in request.files:
            raise ValueError("No files uploaded")
            
        files = request.files.getlist('files')
        if not files or not any(file.filename for file in files):
            raise ValueError("No files selected")
            
        model_type = request.form.get('model_type')
        if not model_type:
            raise ValueError("Model type not selected")
            
        # Validate model type
        valid_models = {
            'gpt-4o': 'OpenAI',
            'gpt-3.5-turbo': 'OpenAI',
            'deepseek-chat': 'Deepseek'
        }
        if model_type not in valid_models:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        role = request.form.get('role')
        if not role:
            raise ValueError("Role not selected")
            
        custom_role = request.form.get('custom_role')
        if role == 'custom' and not custom_role:
            raise ValueError("Custom role description not provided")
            
        logging.info(f"Received request with role: {role}, model: {model_type}, files count: {len(files)}")
        
        # Initialize annotator
        try:
            annotator = PaperAnnotator(
                model_type=model_type,
                input_dir=session_input_dir,
                output_dir=session_output_dir
            )
        except PermissionError as pe:
            logging.error(f"Permission error initializing annotator: {pe}")
            return jsonify({
                'status': 'error',
                'message': f"Permission error: {str(pe)}"
            }), 403
        except Exception as e:
            logging.error(f"Failed to initialize annotator: {e}")
            return jsonify({
                'status': 'error',
                'message': f"Initialization failed: {str(e)}"
            }), 500
            
        # Set role
        if role and role != 'custom':
            annotator.set_role(role)
            logging.info(f"Using predefined role: {role}")
        elif custom_role:
            annotator.set_role(custom_role)
            logging.info("Using custom role")
        else:
            raise ValueError("Must select a role or provide custom role description")
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file and file.filename:
                file_ext = os.path.splitext(file.filename)[1].lower()
                if file_ext not in ['.pdf']:
                    continue
                    
                safe_filename = os.path.join(session_input_dir, os.path.basename(file.filename))
                file.save(safe_filename)
                saved_files.append(safe_filename)
                logging.info(f"Saved file: {file.filename}")
                
                # Send file reception status
                progress_queue.put({
                    'type': 'progress',
                    'file': file.filename,
                    'status': 'reading',
                    'message': f'Processing file: {file.filename}'
                })
        
        if not saved_files:
            raise ValueError("No valid files uploaded")
            
        # Process files one by one
        for file_path in saved_files:
            try:
                # Process single file
                file_name = os.path.basename(file_path)
                result = annotator.process_single_paper(file_path)
                
                if result:
                    # Send processing results
                    progress_queue.put({
                        'type': 'result',
                        'file': file_name,
                        'data': {
                            'file_name': file_name,
                            'visualization_file': result['visualization_file'],
                            'review_data': result['review_data'],
                            'scores': result['scores']
                        },
                        'session_id': session_id
                    })
                else:
                    # Send error message
                    progress_queue.put({
                        'type': 'error',
                        'file': file_name,
                        'message': 'Processing failed'
                    })
                    
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                progress_queue.put({
                    'type': 'error',
                    'file': file_name,
                    'message': str(e)
                })
        
        # Send completion signal
        progress_queue.put({
            'type': 'complete',
            'message': 'All files processed'
        })
        
        return jsonify({'status': 'success', 'session_id': session_id})
            
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logging.error(f"Process error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Processing failed: {str(e)}"
        }), 500

@app.route('/download/<session_id>', methods=['GET'])
def download_results(session_id):
    try:
        # Validate session_id format
        if not re.match(r'^[\w-]+$', session_id):
            logging.error(f"Invalid session ID format: {session_id}")
            return jsonify({'error': 'Invalid session ID format'}), 400
        
        session_output_dir = os.path.join(Config.OUTPUT_DIR, session_id)
        if not os.path.exists(session_output_dir):
            logging.error(f"Output directory not found: {session_output_dir}")
            return jsonify({'error': 'Results directory not found'}), 404
        
        # Create a temporary ZIP file
        zip_path = os.path.join(session_output_dir, 'results.zip')
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in os.listdir(session_output_dir):
                    if file != 'results.zip':  # Do not include ZIP file itself
                        file_path = os.path.join(session_output_dir, file)
                        # Use file name as ZIP path, avoid full path
                        zipf.write(file_path, os.path.basename(file_path))
            
            # Send file
            return send_file(
                zip_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'evaluation_results_{session_id}.zip'
            )
            
        finally:
            # Clean up temporary ZIP file
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception as e:
                    logging.warning(f"Failed to remove temporary zip file: {e}")
        
    except Exception as e:
        logging.error(f"Error in download_results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def progress():
    def generate():
        try:
            # Add a clear message
            yield f"data: {json.dumps({'type': 'clear'})}\n\n"
            
            while True:
                if not progress_queue.empty():
                    progress_data = progress_queue.get()
                    yield f"data: {json.dumps(progress_data)}\n\n"
                time.sleep(0.1)
        except GeneratorExit:
            # Client disconnected
            logging.info("Client disconnected from SSE stream")
        except Exception as e:
            logging.error(f"Error in SSE stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

@app.route('/output/<session_id>/<filename>')
def serve_output_file(session_id, filename):
    try:
        # Decode filename, ensure safe
        filename = os.path.basename(filename)
        file_path = os.path.join(Config.OUTPUT_DIR, session_id, filename)
        
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(
            file_path,
            mimetype='image/png',
            as_attachment=False
        )
    except Exception as e:
        logging.error(f"Error serving file: {e}")
        return jsonify({'error': str(e)}), 500

def send_progress_update(progress_data):
    try:
        # Validate necessary fields
        required_fields = ['file', 'status', 'message']
        if not all(field in progress_data for field in required_fields):
            logging.error(f"Missing required fields in progress data: {progress_data}")
            return
            
        # Send to progress queue
        progress_queue.put(progress_data)
        logging.debug(f"Progress update sent: {progress_data}")
        
    except Exception as e:
        logging.error(f"Error sending progress update: {e}")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

if __name__ == '__main__':
    port = 5001
    try:
        Config.init_dirs()
        cleanup_old_files()
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logging.error(f"Error occurred while starting the program: {e}")
        print(f"Program startup failed: {e}")