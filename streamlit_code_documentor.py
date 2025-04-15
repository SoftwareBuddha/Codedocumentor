import streamlit as st
import os
import sys
import subprocess
import glob
import re
import time
from pathlib import Path
import openai
import logging
import tempfile
import zipfile
import io
import shutil
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('code-documentor')

# Default file types to document
DEFAULT_FILE_EXTENSIONS = [
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c',
    '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt'
]

# Files and directories to ignore by default
DEFAULT_IGNORE_PATTERNS = [
    '*/node_modules/*', '*/venv/*', '*/.git/*', '*/dist/*', '*/build/*',
    '*/.vscode/*', '*/__pycache__/*', '*/vendor/*', '*/.idea/*', '*/bin/*',
    '*/.DS_Store', '*/tmp/*', '*/log/*', '*/.env',
]


class CodeDocumentor:
    def __init__(self,
                 repo_url: str = None,
                 local_repo_path: str = None,
                 output_dir: str = 'docs',
                 api_key: Optional[str] = None,
                 model: str = 'gpt-4',
                 file_extensions: List[str] = None,
                 ignore_patterns: List[str] = None,
                 max_tokens: int = 4000,
                 temperature: float = 0.2,
                 max_files: Optional[int] = None,
                 progress_callback=None):
        """
        Initialize the CodeDocumentor with the provided parameters.

        Args:
            repo_url: URL of the repository to clone and document
            local_repo_path: Path to a local repository (alternative to repo_url)
            output_dir: Directory where documentation will be written
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            model: OpenAI model to use
            file_extensions: List of file extensions to document
            ignore_patterns: List of glob patterns to ignore
            max_tokens: Maximum tokens for API responses
            temperature: Temperature setting for GPT-4 responses
            max_files: Optional limit on the number of files to process
            progress_callback: Function to call with progress updates
        """
        self.repo_url = repo_url
        self.local_repo_path = local_repo_path
        self.output_dir = output_dir
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_files = max_files
        self.progress_callback = progress_callback

        # Set up file extensions to process
        self.file_extensions = file_extensions or DEFAULT_FILE_EXTENSIONS

        # Set up ignore patterns
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS

        # Set OpenAI API key
        if api_key:
            openai.api_key = api_key
        elif os.environ.get('OPENAI_API_KEY'):
            openai.api_key = os.environ.get('OPENAI_API_KEY')
        else:
            raise ValueError(
                "OpenAI API key not provided. Set the OPENAI_API_KEY environment variable or pass via api_key.")

        # Extract repo name
        if repo_url:
            self.repo_name = self._extract_repo_name(repo_url)
            # Path where repo will be cloned
            self.repo_path = f"./{self.repo_name}"
        elif local_repo_path:
            self.repo_path = local_repo_path
            self.repo_name = os.path.basename(local_repo_path)
        else:
            raise ValueError("Either repo_url or local_repo_path must be provided")

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        # Handle various formats of repo URLs
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]

        return repo_url.split('/')[-1]

    def clone_repository(self) -> bool:
        """
        Clone the repository to the local filesystem.

        Returns:
            bool: True if successful, False otherwise
        """
        if self.local_repo_path:
            logger.info(f"Using local repository at {self.local_repo_path}")
            return True

        logger.info(f"Cloning repository: {self.repo_url}")

        try:
            # Check if directory already exists
            if os.path.exists(self.repo_path):
                logger.info(f"Repository directory already exists at {self.repo_path}")
                return True

            # Clone the repo
            process = subprocess.run(['git', 'clone', self.repo_url],
                                     check=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

            logger.info(f"Repository cloned successfully to {self.repo_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            error_message = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
            if self.progress_callback:
                self.progress_callback(f"Failed to clone repository: {error_message}", "error")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during repository cloning: {e}")
            if self.progress_callback:
                self.progress_callback(f"Unexpected error during repository cloning: {str(e)}", "error")
            return False

    def _should_ignore_file(self, file_path: str) -> bool:
        """
        Check if a file should be ignored based on ignore patterns.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if file should be ignored, False otherwise
        """
        # Check against ignore patterns
        for pattern in self.ignore_patterns:
            if glob.fnmatch.fnmatch(file_path, pattern):
                return True

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext not in self.file_extensions

    def find_files_to_document(self) -> List[str]:
        """
        Find all files in the repository that should be documented.

        Returns:
            List[str]: List of file paths to process
        """
        logger.info(f"Finding files to document in {self.repo_path}")

        all_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, start='.')

                if not self._should_ignore_file(rel_path):
                    all_files.append(rel_path)

        logger.info(f"Found {len(all_files)} files to document")

        # Apply max_files limit if set
        if self.max_files and len(all_files) > self.max_files:
            logger.info(f"Limiting to {self.max_files} files as requested")
            return all_files[:self.max_files]

        return all_files

    def read_file_content(self, file_path: str) -> str:
        """
        Read the content of a file, handling various encodings.

        Args:
            file_path: Path to the file to read

        Returns:
            str: Content of the file
        """
        # Try different encodings if UTF-8 fails
        encodings = ['utf-8', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # If all encodings fail, log error and return empty string
        logger.warning(f"Could not decode file {file_path} with any of the attempted encodings")
        return ""

    def _get_documentation_prompt(self, file_path: str, content: str) -> str:
        """
        Generate the prompt for the GPT-4 API to document a file.

        Args:
            file_path: Path to the file
            content: Content of the file

        Returns:
            str: Prompt for GPT-4
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        rel_path = os.path.relpath(file_path, start=self.repo_name if self.repo_url else self.repo_path)

        prompt = f"""
        You are an expert software developer tasked with creating comprehensive documentation for a codebase.

        Please analyze the following code file and create detailed documentation that includes:

        1. A high-level overview of what the file does
        2. The purpose and functionality of each major component (classes, functions, etc.)
        3. Any notable design patterns or architectural approaches used
        4. Important relationships with other parts of the codebase (if apparent)
        5. Any dependencies or requirements
        6. Usage examples where appropriate

        Format your response in Markdown, with appropriate headings, code blocks, and bullet points.

        File Path: {rel_path}
        File Type: {file_ext}

        CODE:
        ```{file_ext[1:] if file_ext else ''}
        {content}
        ```

        Focus on being:
        - Clear and concise
        - Accurate and comprehensive
        - Helpful for both new and experienced developers
        - Consistent with standard documentation practices

        Only include information that can be directly derived from the code. Don't make assumptions beyond what's evident.
        """

        return prompt

    def generate_documentation(self, file_path: str) -> Tuple[str, str]:
        """
        Generate documentation for a single file.

        Args:
            file_path: Path to the file to document

        Returns:
            Tuple[str, str]: The file path for documentation file and the generated documentation
        """
        logger.info(f"Generating documentation for {file_path}")
        if self.progress_callback:
            self.progress_callback(f"Generating documentation for {file_path}")

        try:
            # Read file content
            content = self.read_file_content(file_path)

            if not content:
                logger.warning(f"Empty or unreadable file: {file_path}")
                return file_path, f"# {os.path.basename(file_path)}\n\nUnable to read file content."

            # Build the prompt
            prompt = self._get_documentation_prompt(file_path, content)

            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical documentation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract documentation
            documentation = response.choices[0].message.content.strip()

            return file_path, documentation

        except Exception as e:
            logger.error(f"Error generating documentation for {file_path}: {e}")
            if self.progress_callback:
                self.progress_callback(f"Error generating documentation for {file_path}: {str(e)}", "error")
            return file_path, f"# {os.path.basename(file_path)}\n\nError generating documentation: {str(e)}"

    def save_documentation(self, file_path: str, documentation: str) -> str:
        """
        Save generated documentation to a file.

        Args:
            file_path: Original source file path
            documentation: Generated documentation content

        Returns:
            str: Path to the saved documentation file
        """
        # Create relative path for docs
        if self.repo_url:
            rel_path = os.path.relpath(file_path, start=self.repo_name)
        else:
            rel_path = os.path.relpath(file_path, start=self.repo_path)

        doc_path = os.path.join(self.output_dir, f"{rel_path}.md")

        # Ensure directory exists
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)

        # Write documentation to file
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(documentation)

        logger.info(f"Documentation saved to {doc_path}")
        return doc_path

    def generate_index(self, doc_files: List[str]) -> str:
        """
        Generate an index file for the documentation.

        Args:
            doc_files: List of documentation files

        Returns:
            str: Path to the index file
        """
        logger.info("Generating documentation index")

        # Create the index content
        content = f"# {self.repo_name} Documentation\n\n"
        content += "## Files\n\n"

        # Group files by directory
        file_tree = {}
        for file in doc_files:
            rel_path = os.path.relpath(file, start=self.output_dir)
            if rel_path.endswith('.md'):
                rel_path = rel_path[:-3]  # Remove .md extension

            parts = rel_path.split(os.sep)
            current = file_tree

            # Build tree structure
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # This is a file
                    current.setdefault('files', []).append((part, file))
                else:
                    # This is a directory
                    current.setdefault('dirs', {}).setdefault(part, {})
                    current = current['dirs'][part]

        # Function to recursively build markdown from tree
        def build_markdown(tree, prefix=''):
            result = ''

            # Add directories
            if 'dirs' in tree:
                for name, subtree in sorted(tree['dirs'].items()):
                    result += f"{prefix}- **{name}/**\n"
                    result += build_markdown(subtree, prefix + '  ')

            # Add files
            if 'files' in tree:
                for name, path in sorted(tree['files']):
                    rel_link = os.path.relpath(path, start=self.output_dir)
                    result += f"{prefix}- [{name}]({rel_link})\n"

            return result

        content += build_markdown(file_tree)

        # Add timestamp
        content += f"\n\n---\n*Documentation generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*"

        # Write index file
        index_path = os.path.join(self.output_dir, "index.md")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Index saved to {index_path}")
        return index_path

    def create_documentation_zip(self, output_path='documentation.zip') -> str:
        """
        Create a zip file of the generated documentation.

        Args:
            output_path: Path where the zip file will be saved

        Returns:
            str: Path to the zip file
        """
        logger.info(f"Creating documentation zip file: {output_path}")

        # Check if output directory exists
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Documentation directory not found: {self.output_dir}")

        # Create zip file
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=os.path.dirname(self.output_dir))
                    zipf.write(file_path, arcname)

        logger.info(f"Documentation zip created at {output_path}")
        return output_path

    def run(self) -> Tuple[bool, str]:
        """
        Run the complete documentation process.

        Returns:
            Tuple[bool, str]: Success status and path to the output directory
        """
        try:
            # Clone the repository if needed
            if self.repo_url and not self.clone_repository():
                return False, "Failed to clone repository"

            # Find files to document
            files = self.find_files_to_document()

            if not files:
                logger.warning("No files found to document")
                if self.progress_callback:
                    self.progress_callback("No files found to document", "warning")
                return False, "No files found to document"

            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Generate documentation for each file
            doc_files = []
            total_files = len(files)

            for i, file_path in enumerate(files):
                logger.info(f"Processing file {i + 1}/{total_files}: {file_path}")
                if self.progress_callback:
                    self.progress_callback(f"Processing file {i + 1}/{total_files}: {file_path}",
                                           progress=(i / total_files))

                try:
                    # Generate documentation
                    _, documentation = self.generate_documentation(file_path)

                    # Save documentation
                    doc_path = self.save_documentation(file_path, documentation)
                    doc_files.append(doc_path)

                    # Add a small delay to avoid rate limiting
                    if i < len(files) - 1:
                        time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    if self.progress_callback:
                        self.progress_callback(f"Error processing {file_path}: {str(e)}", "error")

            # Generate index
            index_path = self.generate_index(doc_files)

            if self.progress_callback:
                self.progress_callback("Documentation completed successfully", progress=1.0)

            logger.info(f"Documentation completed successfully. Output in {self.output_dir} directory")
            return True, self.output_dir

        except Exception as e:
            logger.error(f"Unexpected error during documentation process: {e}")
            if self.progress_callback:
                self.progress_callback(f"Unexpected error during documentation process: {str(e)}", "error")
            return False, str(e)


# Streamlit app
def main():
    st.set_page_config(
        page_title="Code Documentor",
        page_icon="📚",
        layout="wide"
    )

    st.title("🚀 Code Documentor")
    st.subheader("Generate comprehensive documentation for your code repositories using GPT-4")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📂 Document from GitHub", "📤 Upload Local Repository"])

    with tab1:
        document_from_github()

    with tab2:
        document_from_upload()


def document_from_github():
    st.header("Document GitHub Repository")

    # Repository URL input
    repo_url = st.text_input("Repository URL", placeholder="https://github.com/username/repository")

    # Git credentials
    st.subheader("Git Authentication (for private repos)")
    auth_type = st.radio("Authentication Method", ["None (Public Repository)", "Username & Password/Token", "SSH Key"])

    if auth_type == "Username & Password/Token":
        git_username = st.text_input("Git Username")
        git_token = st.text_input("Personal Access Token or Password", type="password")

        if git_username and git_token and repo_url:
            # Modify repo URL to include credentials
            if "http://" in repo_url:
                repo_url = repo_url.replace("http://", f"http://{git_username}:{git_token}@")
            elif "https://" in repo_url:
                repo_url = repo_url.replace("https://", f"https://{git_username}:{git_token}@")
            else:
                st.warning("URL must start with http:// or https:// to use username/password authentication")

    elif auth_type == "SSH Key":
        ssh_key = st.text_area("SSH Private Key (content of your id_rsa file)", height=150)
        ssh_password = st.text_input("SSH Key Password (if any)", type="password")

        if ssh_key:
            # Save SSH key to temporary file
            try:
                ssh_dir = os.path.expanduser("~/.ssh")
                os.makedirs(ssh_dir, exist_ok=True)

                with open(os.path.join(ssh_dir, "id_rsa_temp"), "w") as f:
                    f.write(ssh_key)

                # Set correct permissions
                os.chmod(os.path.join(ssh_dir, "id_rsa_temp"), 0o600)

                # Configure Git to use this key
                subprocess.run(
                    ["git", "config", "--global", "core.sshCommand", f"ssh -i {os.path.join(ssh_dir, 'id_rsa_temp')}"])

                st.success("SSH key configured for Git")
            except Exception as e:
                st.error(f"Error configuring SSH key: {str(e)}")

    # Show advanced settings in an expander
    with st.expander("Advanced Settings"):
        openai_api_key = st.text_input("OpenAI API Key", type="password",
                                       help="Your OpenAI API key with access to GPT-4")

        model = st.selectbox("Model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                             help="The OpenAI model to use for generating documentation")

        max_files = st.number_input("Maximum Files", min_value=1, value=50,
                                    help="Maximum number of files to process")

        file_types = st.multiselect("File Types to Document",
                                    ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c',
                                     '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.md', '.html', '.css'],
                                    default=['.py', '.js', '.jsx', '.ts', '.tsx', '.java'])

        ignore_patterns = st.text_area("Ignore Patterns (one per line)",
                                       """*/node_modules/*
  */venv/*
  */.git/*
  */dist/*
  */build/*
  */.vscode/*
  */__pycache__/*
  */vendor/*
  */.idea/*
  */bin/*
  */.DS_Store
  */tmp/*
  */log/*
  */.env""",
                                       help="Glob patterns of files/directories to ignore")

        output_dir = st.text_input("Output Directory", value="docs",
                                   help="Directory where documentation will be written")

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                                help="Temperature for GPT-4 responses (0.0 = deterministic, 1.0 = creative)")

        max_tokens = st.slider("Max Tokens", min_value=1000, max_value=8000, value=4000, step=500,
                               help="Maximum tokens for API responses")

    # Process repository
    if st.button("Generate Documentation", type="primary"):
        if not repo_url:
            st.error("Please enter a repository URL")
            return

        if not openai_api_key and not os.environ.get('OPENAI_API_KEY'):
            st.error("OpenAI API key is required. Please provide it in the Advanced Settings.")
            return

        # Set up progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(message, progress=None, status="info"):
            if progress is not None:
                progress_bar.progress(progress)

            if status == "error":
                status_text.error(message)
            elif status == "warning":
                status_text.warning(message)
            else:
                status_text.info(message)

        # Process ignore patterns
        ignore_patterns_list = [pattern.strip() for pattern in ignore_patterns.split("\n") if pattern.strip()]

        try:
            # Create a temporary directory for the output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize the documentor
                documentor = CodeDocumentor(
                    repo_url=repo_url,
                    output_dir=output_dir if output_dir else "docs",
                    api_key=openai_api_key if openai_api_key else None,
                    model=model,
                    file_extensions=file_types,
                    ignore_patterns=ignore_patterns_list,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_files=max_files,
                    progress_callback=update_progress
                )

                # Run the documentation process
                success, result = documentor.run()

                if success:
                    # Create a zip file of the documentation
                    zip_path = os.path.join(temp_dir, "documentation.zip")
                    documentor.create_documentation_zip(zip_path)

                    # Read the zip file for download
                    with open(zip_path, "rb") as f:
                        zip_data = f.read()

                    # Display success message and download button
                    st.success(f"Documentation generated successfully! Output in {result} directory.")
                    st.download_button(
                        label="Download Documentation (ZIP)",
                        data=zip_data,
                        file_name="documentation.zip",
                        mime="application/zip"
                    )

                    # Show the index file
                    index_path = os.path.join(result, "index.md")
                    if os.path.exists(index_path):
                        with open(index_path, "r") as f:
                            index_content = f.read()
                        st.markdown("## Documentation Index")
                        st.markdown(index_content)
                else:
                    st.error(f"Documentation generation failed: {result}")

        except Exception as e:
            st.error(f"Error during documentation process: {str(e)}")


def document_from_upload():
    st.header("Document Local Repository")

    # File uploader for ZIP file
    uploaded_file = st.file_uploader("Upload Repository ZIP", type="zip",
                                     help="Upload a ZIP file containing your source code repository")

    # Show advanced settings in an expander
    with st.expander("Advanced Settings"):
        openai_api_key = st.text_input("OpenAI API Key (Upload)", type="password",
                                       help="Your OpenAI API key with access to GPT-4")

        model = st.selectbox("Model (Upload)", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                             help="The OpenAI model to use for generating documentation")

        max_files = st.number_input("Maximum Files (Upload)", min_value=1, value=50,
                                    help="Maximum number of files to process")

        file_types = st.multiselect("File Types to Document (Upload)",
                                    ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c',
                                     '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.md', '.html', '.css'],
                                    default=['.py', '.js', '.jsx', '.ts', '.tsx', '.java'])

        ignore_patterns = st.text_area("Ignore Patterns (Upload) (one per line)",
                                       """*/node_modules/*
  */venv/*
  */.git/*
  */dist/*
  */build/*
  */.vscode/*
  */__pycache__/*
  */vendor/*
  */.idea/*
  */bin/*
  */.DS_Store
  */tmp/*
  */log/*
  */.env""",
                                       help="Glob patterns of files/directories to ignore")

        output_dir = st.text_input("Output Directory (Upload)", value="docs",
                                   help="Directory where documentation will be written")

        temperature = st.slider("Temperature (Upload)", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                                help="Temperature for GPT-4 responses (0.0 = deterministic, 1.0 = creative)")

        max_tokens = st.slider("Max Tokens (Upload)", min_value=1000, max_value=8000, value=4000, step=500,
                               help="Maximum tokens for API responses")

    # Process uploaded repository
    if uploaded_file is not None and st.button("Generate Documentation from Upload", type="primary"):
        if not openai_api_key and not os.environ.get('OPENAI_API_KEY'):
            st.error("OpenAI API key is required. Please provide it in the Advanced Settings.")
            return

        # Set up progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(message, progress=None, status="info"):
            if progress is not None:
                progress_bar.progress(progress)

            if status == "error":
                status_text.error(message)
            elif status == "warning":
                status_text.warning(message)
            else:
                status_text.info(message)

        # Process ignore patterns
        ignore_patterns_list = [pattern.strip() for pattern in ignore_patterns.split("\n") if pattern.strip()]

        try:
            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_extract_dir:
                # Save uploaded file
                zip_path = os.path.join(temp_extract_dir, "repo.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)

                # Find the repository root directory (assuming it's the only directory or the zip file itself)
                extracted_items = [os.path.join(temp_extract_dir, item) for item in os.listdir(temp_extract_dir)
                                   if item != "repo.zip"]

                repo_root = None
                for item in extracted_items:
                    if os.path.isdir(item):
                        repo_root = item
                        break

                # If we didn't find a directory, use the extract dir itself
                if not repo_root:
                    repo_root = temp_extract_dir

                # Create output directory
                output_path = os.path.join(temp_extract_dir, output_dir)
                os.makedirs(output_path, exist_ok=True)

                update_progress(f"Processing repository from uploaded ZIP file", progress=0.1)

                # Initialize the documentor
                documentor = CodeDocumentor(
                    local_repo_path=repo_root,
                    output_dir=output_path,
                    api_key=openai_api_key if openai_api_key else None,
                    model=model,
                    file_extensions=file_types,
                    ignore_patterns=ignore_patterns_list,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_files=max_files,
                    progress_callback=update_progress
                )

                # Run the documentation process
                success, result = documentor.run()

                if success:
                    # Create a zip file of the documentation
                    zip_path = os.path.join(temp_extract_dir, "documentation.zip")
                    documentor.create_documentation_zip(zip_path)

                    # Read the zip file for download
                    with open(zip_path, "rb") as f:
                        zip_data = f.read()

                    # Display success message and download button
                    st.success(f"Documentation generated successfully!")
                    st.download_button(
                        label="Download Documentation (ZIP)",
                        data=zip_data,
                        file_name="documentation.zip",
                        mime="application/zip"
                    )

                    # Show the index file
                    index_path = os.path.join(output_path, "index.md")
                    if os.path.exists(index_path):
                        with open(index_path, "r") as f:
                            index_content = f.read()
                        st.markdown("## Documentation Index")
                        st.markdown(index_content)
                else:
                    st.error(f"Documentation generation failed: {result}")

        except Exception as e:
            st.error(f"Error during documentation process: {str(e)}")


if __name__ == "__main__":
    main()