from queue import Queue

# Create global progress queue
progress_queue = Queue()

def send_progress_update(progress_data):
    """
    Send progress update to the progress queue.
    
    Args:
        progress_data (dict): Dictionary containing progress information
    """
    try:
        progress_queue.put(progress_data)
    except Exception as e:
        import logging
        logging.error(f"Failed to send progress update: {e}")