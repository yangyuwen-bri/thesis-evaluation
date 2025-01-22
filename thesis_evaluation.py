import os
import pandas as pd
from PyPDF2 import PdfReader
import httpx
from openai import OpenAI
import tiktoken
import logging
import shutil
import time
from datetime import datetime, timedelta
from utils.progress import send_progress_update
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import json
import re

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("thesis_evaluation.log"),
        logging.StreamHandler()  # Also print to console
    ]
)

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    INPUT_DIR = os.path.join(TEMP_DIR, 'input')
    OUTPUT_DIR = os.path.join(TEMP_DIR, 'output')
    CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')
    
    @classmethod
    def init_dirs(cls):
        """Initialize required directories"""
        for dir_path in [cls.TEMP_DIR, cls.INPUT_DIR, cls.OUTPUT_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def cleanup_temp(cls):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(cls.TEMP_DIR):
            shutil.rmtree(cls.TEMP_DIR)
    
    @classmethod
    def load_config(cls):
        """Load configuration file"""
        try:
            with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")
            raise

class ProcessStatus:
    UPLOADING = 'uploading'    # File uploading
    READING = 'reading'        # Reading file
    ANALYZING = 'analyzing'    # Analyzing content
    PROCESSING = 'processing'  # Processing results
    COMPLETED = 'completed'    # Processing completed
    ERROR = 'error'           # Error occurred

    @staticmethod
    def update_progress(file_id, status, message=None):
        progress_data = {
            'file': file_id,
            'status': status,
            'message': message
        }
        # Send to progress notification queue

class PaperAnnotator:
    roles = {
        "computer_science": ("Main research areas include web search, information retrieval, e-commerce, etc., with about 380 research publications.", "Professor of Computer Science"),
        "library_science": ("Research interests in library and information service management, with a focus on the impact of public libraries.", "Professor of Library Science"),
        "info_management": ("Research interests include IT usage motivation, misinformation dissemination, etc., served as associate editor for several important journals.", "Professor of Information Management"),
        "cybersecurity": ("Professor of Cyberspace Content Security", "Professor of Cyberspace Content Security"),
        "economics": ("Professor of Economics", "Professor of Economics"),
        "journalism": ("Research focuses on international communication, intelligent communication, and the impact of new media environments on communication effects and social influence.", "Professor of Journalism")
    }

    # Modify model configurations
    MODEL_CONFIGS = {
        'gpt': {
            'temperature': 0.2,
            'top_p': 1
        },
        'deepseek': {
            'temperature': 0.2,
            'top_p': 1
        }
    }

    # Store prompt templates separately
    PROMPT_TEMPLATES = {
        'gpt': {
            'system': """You are a professional academic paper reviewer, acting as {role_name}. {role_description}
            You need to carefully read the paper and extract relevant evidence to support your evaluation.
            
            As {role_name}, you should pay special attention to:
            1. The academic value and innovation of the paper
            2. The standardization and rationality of the research methods
            3. The reliability and contribution of the paper's conclusions
            4. The logic and standardization of the writing
            """
        },
        'deepseek': {
            'system': """You are now a senior academic paper review expert, acting as {role_name}. {role_description}
            Please conduct a comprehensive and detailed evaluation of the paper based on your professional background. During the evaluation, you should:
            1. Extract specific evidence from the paper to support your scoring
            2. Maintain objectivity and professionalism in your evaluation
            3. Ensure that the scoring reasons are detailed and specific
            4. Provide constructive improvement suggestions
            
            As {role_name}, focus on:
            1. The theoretical innovation and academic contribution of the paper
            2. The scientific and standardized nature of the research methods
            3. The reliability and value of the research conclusions
            4. The standardization and logic of academic writing
            """
        }
    }

    def __init__(self, model_type, input_dir=None, output_dir=None):
        self.input_dir = input_dir or Config.INPUT_DIR
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.model_type = model_type
        
        # Determine model provider
        self.model_provider = 'deepseek' if 'deepseek' in model_type.lower() else 'gpt'
        
        # Get model configuration
        self.model_config = self.MODEL_CONFIGS[self.model_provider]
        
        # Validate directories
        self.validate_dirs()
        
        # Load configuration
        config = Config.load_config()
        
        # Select configuration based on model type
        if self.model_provider == 'deepseek':
            base_url = config['deepseek_api_base_url']
            api_key = config['deepseek_api_key']
        else:
            base_url = config['api_base_url']
            api_key = config['api_key']
            
        # Initialize client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True,
            ),
        )
        
        # Add font path
        self.font_path = Path(__file__).parent / 'fonts' / 'SourceHanSansCN-Normal.otf'
        
        # Define evaluation standards file path
        self.standards_file = Path(__file__).parent / 'standards' / 'evaluation_standards.xlsx'
        
        # Load dimensions and standards from evaluation standards file
        standards_data = self._load_evaluation_standards()
        if not standards_data:
            raise ValueError("Failed to load evaluation standards file, please check if the file format is correct")
        else:
            self.evaluation_standards, self.dimensions = standards_data

        self.selected_role = None
    
    def validate_dirs(self):
        """
        Validate input and output directories.
        
        This method checks if directories exist and have proper permissions.
        Creates directories if they do not exist.
        Raises PermissionError if read/write access is denied.
        """
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Check directory permissions
        if not os.access(self.input_dir, os.R_OK):
            raise PermissionError(f"No read permission for input directory: {self.input_dir}")
        
        if not os.access(self.output_dir, os.W_OK):
            raise PermissionError(f"No write permission for output directory: {self.output_dir}")

    def set_role(self, role_key):
        """Set evaluation role"""
        if role_key in self.roles:
            self.selected_role = self.roles[role_key]
        else:
            # For custom roles, use role description as name
            self.selected_role = (role_key, role_key)  # Modify here, use role_key as description

    def extract_full_content(self, file_path):
        try:
            pdf_reader = PdfReader(file_path)
            full_content = ""
            for page in pdf_reader.pages:
                full_content += page.extract_text()

            if not full_content.strip():
                logging.warning(f"Unable to extract content from file: {file_path}")
            return full_content
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return ""

    def _load_evaluation_standards(self):
        """Load evaluation standards from Excel file"""
        try:
            if not self.standards_file.exists():
                logging.error(f"Evaluation standards file does not exist: {self.standards_file}")
                return None
            
            # Read Excel file
            standards_df = pd.read_excel(self.standards_file)
            standards = {}
            
            # Verify necessary columns exist
            required_columns = ['Evaluation Dimension', 'Scoring Criteria']
            if not all(col in standards_df.columns for col in required_columns):
                logging.error("Evaluation standards file format error: missing necessary columns")
                return None
            
            # Get dimension list from file
            dimensions = standards_df['Evaluation Dimension'].tolist()
            
            # Convert DataFrame to dictionary
            for _, row in standards_df.iterrows():
                dimension = row['Evaluation Dimension']
                standards[dimension] = row['Scoring Criteria']
            
            return standards, dimensions
        except Exception as e:
            logging.error(f"Error loading evaluation standards: {e}")
            return None

    def _get_system_prompt(self, role_name, role_description):
        """Get system prompt"""
        template = self.PROMPT_TEMPLATES[self.model_provider]['system']
        return template.format(
            role_name=role_name,
            role_description=role_description
        )

    def classify_paper(self, file_name, content, custom_role=None):
        """Evaluate paper content and generate scores and comments"""
        try:
            if custom_role:
                role_name = custom_role
                role_description = "Custom role description"
            else:
                role_name, role_description = self.selected_role

            # Get system prompt
            system_message = self._get_system_prompt(role_name, role_description)

            # Build dimension example string
            dimensions_examples = {}
            for dim in self.dimensions:
                dimensions_examples[dim] = self.evaluation_standards.get(dim, "No scoring criteria provided")
            
            # Build dimension list string
            dimensions_list = "\n".join([f"- {dim}: {standard}" 
                                       for dim, standard in dimensions_examples.items()])

            # Build scoring reason template for each dimension
            reason_template = (
                '"Score: X points\n'
                'Paper excerpt: [Accurately quote relevant paragraphs from the paper]\n'
                'Scoring reason: [Detailed analysis]\n'
                'Scoring basis: 1. ... 2. ... 3. ..."'
            )
            
            # Build dimension scoring reason string
            reasons_str = ', '.join([f'"{dim}": {reason_template}' for dim in self.dimensions])
            scores_str = ', '.join([f'"{dim}": score' for dim in self.dimensions])

            # Optimize prompt with stricter format requirements
            prompt = f"""
            Please professionally evaluate the following paper:

            ### File Name
            {file_name}

            ### Evaluation Dimensions and Criteria
            {dimensions_list}

            ### Paper Content
            {content}

            ### Output Requirements
            Please strictly follow the JSON format below for the evaluation results, which must include and only include the following {len(self.dimensions)} dimensions.
            Each dimension's scoring reason must include four parts: score, paper excerpt, scoring reason, and scoring basis.

            {{
                "scores": {{
                    {scores_str}
                }},
                "reasons": {{
                    {reasons_str}
                }}
            }}

            Notes:
            1. Strictly use the given dimension names, do not change or add other dimensions
            2. Each dimension's score must be within the 0-10 range
            3. Scoring reasons must include paper excerpts as evidence
            4. Scoring reasons should be objective and professional, avoid subjective judgments
            5. Scoring basis should be specific and actionable
            """

            try:
                logging.info(f"Starting to process file: {file_name}")
                # Get model parameters from configuration
                model_params = self.MODEL_CONFIGS[self.model_provider].copy()
                
                response = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    **model_params  # Only pass valid parameters
                )
                
                if not response or not response.choices:
                    logging.error(f"API returned invalid response: {response}")
                    return None

                result_text = response.choices[0].message.content.strip()
                logging.debug(f"API raw response: {result_text}")
                
                # Clean and parse JSON
                cleaned_text = self._clean_json_text(result_text)
                result_data = json.loads(cleaned_text)
                
                # Add strict response validation
                if not self._validate_response(result_data):
                    logging.error("Response data validation failed")
                    return None
                
                # Save scoring reasons
                reasons_path = Path(self.output_dir) / f"{file_name}_scoring_reasons.txt"
                try:
                    with open(reasons_path, 'w', encoding='utf-8') as f:
                        for dim in self.dimensions:
                            f.write(f"═══════════ {dim} ═══════════\n\n")
                            reason = result_data['reasons'].get(dim, 'No scoring reason provided')
                            # Ensure scoring reason format is correct
                            if not reason.startswith("Score:"):
                                reason = f"Score: {result_data['scores'].get(dim, 0)} points\n{reason}"
                            f.write(f"{reason}\n\n")
                            f.write("═══════════════════════════════════\n\n")
                    logging.info(f"Scoring reasons saved to: {reasons_path}")
                except Exception as e:
                    logging.error(f"Error saving scoring reasons: {e}")
                
                # Extract scores
                scores = [result_data['scores'].get(dim, 0) for dim in self.dimensions]
                
                # Generate review comments
                review_result = self.generate_review_comments(str(reasons_path))
                
                return {
                    'scores': scores,
                    'review_data': review_result.get('data') if review_result else None
                }
                
            except Exception as e:
                logging.error(f"Error processing file: {e}")
                return None
                
        except Exception as e:
            logging.error(f"API call error: {e}")
            return None

    def _validate_response(self, result_data):
        """Validate API response data integrity and correctness"""
        try:
            # Check basic structure
            if not isinstance(result_data, dict):
                logging.error("Response data is not in dictionary format")
                return False
            
            if 'scores' not in result_data or 'reasons' not in result_data:
                logging.error("Response data missing necessary keys")
                return False
            
            # Validate dimension integrity
            for dim in self.dimensions:
                # Check score
                if dim not in result_data['scores']:
                    logging.error(f"Missing score for dimension: {dim}")
                    return False
                
                score = result_data['scores'][dim]
                if not isinstance(score, (int, float)) or score < 0 or score > 10:
                    logging.error(f"Invalid score for dimension {dim}: {score}")
                    return False
                
                # Check scoring reason
                if dim not in result_data['reasons']:
                    logging.error(f"Missing scoring reason for dimension: {dim}")
                    return False
                
                reason = result_data['reasons'][dim]
                if not isinstance(reason, str) or not reason.strip():
                    logging.error(f"Invalid scoring reason for dimension {dim}")
                    return False
                
                # Check scoring reason format
                required_parts = ['Score:', 'Paper excerpt:', 'Scoring reason:', 'Scoring basis:']
                if not all(part in reason for part in required_parts):
                    logging.error(f"Incomplete scoring reason format for dimension {dim}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating response data: {e}")
            return False

    def process_papers(self, file_type, custom_role=None):
        try:
            files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(f'.{file_type}')]
            total_files = len(files)
            processed_files = []
            
            if not files:
                logging.warning("No files found to process.")
                return None
            
            for index, file_name in enumerate(files):
                try:
                    # Send status update: start reading
                    send_progress_update({
                        'file': file_name,
                        'status': ProcessStatus.READING,
                        'current': index + 1,
                        'total': total_files,
                        'message': f'Reading file: {file_name}'
                    })
                    
                    file_path = os.path.join(self.input_dir, file_name)
                    content = self.extract_full_content(file_path)
                    
                    if not content.strip():
                        send_progress_update({
                            'file': file_name,
                            'status': ProcessStatus.ERROR,
                            'current': index + 1,
                            'total': total_files,
                            'message': 'File content is empty'
                        })
                        continue
                    
                    # Send status update: start analysis
                    send_progress_update({
                        'file': file_name,
                        'status': ProcessStatus.ANALYZING,
                        'current': index + 1,
                        'total': total_files,
                        'message': f'Analyzing file: {file_name}'
                    })
                    
                    result = self.classify_paper(file_name, content, custom_role)
                    
                    if not result or 'scores' not in result:
                        send_progress_update({
                            'file': file_name,
                            'status': ProcessStatus.ERROR,
                            'current': index + 1,
                            'total': total_files,
                            'message': 'Failed to generate scores'
                        })
                        continue
                    
                    # Create visualization
                    visualization_file = self._create_visualization(
                        pd.DataFrame({
                            'Evaluation Dimension': self.dimensions,
                            'Score': result['scores']
                        }), 
                        self.selected_role[1],
                        Path(self.output_dir),
                        file_name
                    )
                    
                    if visualization_file:
                        processed_files.append({
                            'file_name': file_name,
                            'scores': result['scores'],
                            'visualization_file': visualization_file,
                            'review_data': result.get('review_data')
                        })
                        # Send completion status
                        send_progress_update({
                            'file': file_name,
                            'status': ProcessStatus.COMPLETED,
                            'current': index + 1,
                            'total': total_files,
                            'message': f'Processing completed: {file_name}'
                        })
                    
                except Exception as e:
                    logging.error(f"Error processing file {file_name}: {str(e)}")
                    send_progress_update({
                        'file': file_name,
                        'status': ProcessStatus.ERROR,
                        'current': index + 1,
                        'total': total_files,
                        'message': str(e)
                    })
                    continue
            
            if not processed_files:
                raise ValueError("No files were successfully processed")
            
            return {
                'files': processed_files,
                'scores_file': os.path.join(self.output_dir, f"annotated_results_{self.selected_role[1].replace(' ', '_')}.xlsx")
            }
            
        except Exception as e:
            logging.error(f"Process error: {str(e)}")
            raise  # Propagate error instead of returning None

    def get_review_data(self, review_file):
        """Read review data"""
        try:
            if os.path.exists(review_file):
                with open(review_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Failed to read review data: {e}")
        return None

    def preview_pdf_content(self, file_path):
        """Preview PDF content and save as a temporary txt file"""
        content = self.extract_full_content(file_path)
        
        # Basic statistics
        total_chars = len(content)
        total_lines = len(content.splitlines())
        
        # Print preview information
        print(f"\n{'='*50}")
        print(f"PDF file: {os.path.basename(file_path)}")
        print(f"Total characters: {total_chars}")
        print(f"Total lines: {total_lines}")
        print(f"\nContent preview (first 500 characters):")
        print(f"{content[:500]}...")
        print(f"\n{'='*50}")
        
        # Save as temporary txt file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        txt_path = os.path.join(self.output_dir, f"{base_name}_extracted.txt")
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"File name: {os.path.basename(file_path)}\n")
                f.write(f"Total characters: {total_chars}\n")
                f.write(f"Total lines: {total_lines}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(content)
            logging.info(f"PDF content saved to: {txt_path}")
        except Exception as e:
            logging.error(f"Error saving PDF content to txt file: {e}")
        
        return content

    def _create_visualization(self, scores_df, role_name, output_dir, file_name=None):
        try:
            # Set Matplotlib to use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            
            # Generate unique file name
            if file_name:
                # Remove file extension, replace invalid characters
                base_name = os.path.splitext(file_name)[0]
                safe_name = re.sub(r'[^\w\-_\.]', '_', base_name)
                viz_filename = f"{role_name}_{safe_name}_score_visualization.png"
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                viz_filename = f"{role_name}_{timestamp}_score_visualization.png"
            
            # Ensure output path is a Path object
            if isinstance(output_dir, str):
                output_dir = Path(output_dir)
            
            # Create full output path
            output_path = output_dir / viz_filename
            
            # Create chart
            plt.figure(figsize=(20, 12))
            
            # Set background color
            plt.gca().set_facecolor('#f8f9fa')
            plt.gcf().patch.set_facecolor('#f8f9fa')
            
            # Use seaborn's Set3 color scheme
            colors = sns.color_palette("Set3", len(scores_df))
            
            # Create bar chart
            x = range(len(scores_df))
            bars = plt.gca().bar(x, scores_df['Score'], 
                         color=colors,
                         edgecolor='white',
                         linewidth=1.5)
            
            # Set x-axis ticks and labels
            plt.gca().set_xticks(x)  # Set tick positions
            
            # Set labels, add spacing, and adjust angle
            if self.font_path.exists():
                font = FontProperties(fname=str(self.font_path))
                plt.gca().set_xticks(x, 
                          scores_df['Evaluation Dimension'], 
                          rotation=45,
                          ha='right',
                          fontproperties=font,
                          fontsize=10)
                
                plt.gca().set_title(f'{role_name} Paper Evaluation Results', 
                            fontproperties=font,
                            fontsize=16,
                            pad=20)
                plt.gca().set_xlabel('Evaluation Dimension', 
                            fontproperties=font,
                            fontsize=12,
                            labelpad=15)
                plt.gca().set_ylabel('Score',
                            fontproperties=font,
                            fontsize=12)
            else:
                plt.gca().set_xticks(x, 
                          scores_df['Evaluation Dimension'], 
                          rotation=45,
                          ha='right',
                          fontsize=10)
            
            # Set y-axis range and grid lines
            plt.gca().set_ylim(0, 10.5)
            plt.gca().grid(True, linestyle='--', alpha=0.3, color='gray', axis='y')
            
            # Set axis style
            plt.gca().spines['bottom'].set_color('#cccccc')
            plt.gca().spines['left'].set_color('#cccccc')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.gca().text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center',
                       va='bottom',
                       fontsize=10,
                       color='#333333')
            
            # Adjust layout to ensure all labels are visible
            plt.subplots_adjust(bottom=0.25)  # Increase bottom margin
            plt.tight_layout()
            
            # Save chart
            plt.savefig(str(output_path),  # Convert to string
                       dpi=300,
                       bbox_inches='tight',
                       pad_inches=0.5,
                       facecolor='#f8f9fa')
            
            plt.close()
            
            # Verify file was created successfully
            if not output_path.exists():
                logging.error(f"Failed to create visualization file: {output_path}")
                return None
            
            logging.info(f"Successfully created visualization: {viz_filename}")
            return viz_filename
            
        except Exception as e:
            logging.error(f"Error creating visualization: {e}")
            logging.error(f"Error details: {str(e)}")
            return None

    def generate_review_comments(self, reasons_file_path):
        """Generate review comments based on scoring reasons"""
        try:
            with open(reasons_file_path, 'r', encoding='utf-8') as f:
                reasons_text = f.read()
            
            system_message = """You are an experienced journal editor, skilled in providing professional, objective, and detailed comments on academic papers.
            For papers in different fields, you need to combine the theories, methods, and latest perspectives of the discipline to provide evaluations.
            
            Please write a structured, clear, and constructive review based on the scoring reasons, providing specific optimization suggestions. Avoid vague expressions such as "some methods are not rigorous" or "some theories are not solid." Suggestions should be as detailed, actionable, and professional as possible.
            The review should be output in JSON format."""
            
            prompt = f"""
            Please write a review and suggestions based on the following scoring reasons. Point out specific methods, perspectives, theories, and provide detailed, actionable suggestions:

            {reasons_text}

            Review requirements:
            1. Overall evaluation (200 words), starting with "This paper..."
            2. Specific evaluation in 5 points: paper topic, research methods, research content, core findings, writing standards
            3. Each point should be limited to 100 words
            4. Highlight the paper's core strengths, existing issues, and improvement suggestions
            5. Language should be concise, professional, and detailed, avoiding generalizations

            Please strictly follow the JSON format below for output. Do not include any other content:
            {{
                "overall": "Overall evaluation content",
                "details": {{
                    "Paper Topic": "Evaluation of paper topic",
                    "Research Methods": "Evaluation of research methods",
                    "Research Content": "Evaluation of research content",
                    "Core Findings": "Evaluation of core findings",
                    "Writing Standards": "Evaluation of writing standards"
                }}
            }}
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    top_p=1
                )
                
                if response and response.choices:
                    review_json = response.choices[0].message.content.strip()
                    
                    # Clean JSON text
                    review_json = self._clean_json_text(review_json)
                    
                    try:
                        review_data = json.loads(review_json)
                        
                        if 'overall' not in review_data or 'details' not in review_data:
                            raise ValueError("Missing required keys in JSON response")
                        
                        standardized_details = {
                            "Topic": review_data['details'].get('Paper Topic', 'No review available'),
                            "Methodology": review_data['details'].get('Research Methods', 'No review available'),
                            "Content": review_data['details'].get('Research Content', 'No review available'),
                            "Findings": review_data['details'].get('Core Findings', 'No review available'),
                            "Writing": review_data['details'].get('Writing Standards', 'No review available')
                        }
                        
                        review_data['details'] = standardized_details
                        
                        formatted_review = (
                            "Overall Evaluation:\n"
                            f"{review_data['overall']}\n\n"
                            "Specific Evaluation:\n"
                            f"1. Paper Topic:\n{review_data['details']['Topic']}\n\n"
                            f"2. Research Methods:\n{review_data['details']['Methodology']}\n\n"
                            f"3. Research Content:\n{review_data['details']['Content']}\n\n"
                            f"4. Core Findings:\n{review_data['details']['Findings']}\n\n"
                            f"5. Writing Standards:\n{review_data['details']['Writing']}\n"
                        )
                        
                        output_dir = Path(reasons_file_path).parent
                        review_path = output_dir / f"{Path(reasons_file_path).stem}_review.txt"
                        json_path = output_dir / f"{Path(reasons_file_path).stem}_review.json"
                        
                        with open(review_path, 'w', encoding='utf-8') as f:
                            f.write(formatted_review)
                        
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(review_data, f, ensure_ascii=False, indent=2)
                        
                        return {
                            'text_file': str(review_path),
                            'json_file': str(json_path),
                            'data': review_data
                        }
                        
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parsing error: {e}")
                        logging.error(f"Problematic JSON string: {review_json}")
                        return None
                        
                return None
                
            except Exception as e:
                logging.error(f"Error generating review: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Error reading reasons file: {e}")
            return None

    def _clean_json_text(self, text):
        """Clean JSON text, handle line breaks and control characters"""
        try:
            # Remove possible code block markers
            text = text.strip()
            if text.startswith('```json'):
                text = text.replace('```json', '', 1)
            if text.endswith('```'):
                text = text.replace('```', '', 1)
            text = text.strip()
            
            # Try to directly parse JSON
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                # 1. Normalize line breaks
                text = text.replace('\r\n', '\n').replace('\r', '\n')
                
                # 2. Handle formatting issues in scoring reasons
                text = re.sub(r'\n\s+', '\n', text)  # Remove spaces after line breaks
                
                # 3. Clean all control characters, but keep basic line breaks
                text = ''.join(char for char in text if (
                    ord(char) >= 32 or char == '\n'
                ))
                
                # 4. Handle quote issues
                text = text.replace('"', '"').replace('"', '"')
                text = text.replace("'", '"')  # Unify to double quotes
                
                # 5. Normalize JSON format
                text = re.sub(r',\s*}', '}', text)  # Remove extra commas at the end of objects
                text = re.sub(r',\s*]', ']', text)  # Remove extra commas at the end of arrays
                text = re.sub(r'\s+(?=[\{\}\[\],])', '', text)  # Remove whitespace before punctuation
                
                # 6. Handle extra line breaks
                text = re.sub(r'\n+', '\n', text)
                
                # 7. Validate cleaned text
                try:
                    json.loads(text)
                    logging.info("JSON cleaning successful")
                    return text
                except json.JSONDecodeError as e:
                    logging.error(f"JSON still invalid after cleaning: {e}")
                    logging.error(f"Cleaned text: {text}")
                    raise
                
        except Exception as e:
            logging.error(f"Error cleaning JSON text: {e}")
            raise

    def process_single_paper(self, file_path):
        """Process a single paper file"""
        try:
            file_name = os.path.basename(file_path)
            logging.info(f"Processing single paper: {file_name}")
            
            # Send status update: start reading
            send_progress_update({
                'type': 'progress',
                'file': file_name,
                'status': ProcessStatus.READING,
                'message': f'Reading file: {file_name}'
            })
            
            # Extract text content
            content = self.extract_full_content(file_path)
            if not content:
                raise ValueError("Unable to extract file content")
            
            # Send status update: start analysis
            send_progress_update({
                'type': 'progress',
                'file': file_name,
                'status': ProcessStatus.ANALYZING,
                'message': f'Analyzing content: {file_name}'
            })
            
            # Analyze paper
            result = self.classify_paper(file_name, content)
            if not result:
                raise ValueError("Failed to analyze paper")
            
            # Send status update: start generating visualization
            send_progress_update({
                'type': 'progress',
                'file': file_name,
                'status': ProcessStatus.PROCESSING,
                'message': f'Generating visualization: {file_name}'
            })
            
            # Create visualization
            visualization_file = self._create_visualization(
                pd.DataFrame({
                    'Evaluation Dimension': self.dimensions,
                    'Score': result['scores']
                }), 
                self.selected_role[1],
                Path(self.output_dir),
                file_name
            )
            
            if not visualization_file:
                raise ValueError("Failed to generate visualization")
            
            # Build return result
            processed_result = {
                'file_name': file_name,
                'scores': result['scores'],
                'visualization_file': visualization_file,
                'review_data': result.get('review_data')
            }
            
            # Send completion status
            send_progress_update({
                'type': 'progress',
                'file': file_name,
                'status': ProcessStatus.COMPLETED,
                'message': f'Processing completed: {file_name}'
            })
            
            return processed_result
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            # Send error status
            send_progress_update({
                'type': 'progress',
                'file': file_name,
                'status': ProcessStatus.ERROR,
                'message': str(e)
            })
            return None

def cleanup_old_files():
    """Clean up temporary files older than 24 hours"""
    cutoff = datetime.now() - timedelta(hours=24)
    
    for root, dirs, files in os.walk(Config.TEMP_DIR):
        for name in files + dirs:
            path = os.path.join(root, name)
            if os.path.getctime(path) < cutoff.timestamp():
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                except Exception as e:
                    logging.warning(f"Failed to clean up old file {path}: {e}")

if __name__ == "__main__":
    try:
        from app import app
        Config.init_dirs()
        cleanup_old_files()
        app.run(host='0.0.0.0', port=5001, debug=True)  # Use port 5001
    except Exception as e:
        logging.error(f"Error starting the program: {e}")
        print(f"Program failed to start: {e}")
