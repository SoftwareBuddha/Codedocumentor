#!/usr/bin/env python3
"""
Code Documentor - Automatically generate documentation for a repository using GPT-4

This script:
1. Clones a GitHub repository
2. Analyzes the code files
3. Uses GPT-4 to generate documentation
4. Outputs documentation as Markdown files in a docs/ directory
"""

import os
import sys
import argparse
import subprocess
import glob
import re
import time
from pathlib import Path
import openai
import logging
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
                 repo_url: str, 
                 output_dir: str = 'docs',
                 api_key: Optional[str] = None,
                 model: str = 'gpt-4',
                 file_extensions: List[str] = None,
                 ignore_patterns: List[str] = None,
                 max_tokens: int = 4000,
                 temperature: float = 0.2,
                 max_files: Optional[int] = None):
        """
        Initialize the CodeDocumentor with the provided parameters.
        
        Args:
            repo_url: URL of the repository to clone and document
            output_dir: Directory where documentation will be written
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            model: OpenAI model to use
            file_extensions: List of file extensions to document
            ignore_patterns: List of glob patterns to ignore
            max_tokens: Maximum tokens for API responses
            temperature: Temperature setting for GPT-4 responses
            max_files: Optional limit on the number of files to process
        """
        self.repo_url = repo_url
        self.output_dir = output_dir
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_files = max_files
        
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
            raise ValueError("OpenAI API key not provided. Set the OPENAI_API_KEY environment variable or pass via --api-key.")

        # Extract repo name from URL
        self.repo_name = self._extract_repo_name(repo_url)
        
        # Path where repo will be cloned
        self.repo_path = f"./{self.repo_name}"
        
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
        logger.info(f"Cloning repository: {self.repo_url}")
        
        try:
            # Check if directory already exists
            if os.path.exists(self.repo_path):
                logger.info(f"Repository directory already exists at {self.repo_path}")
                return True
                
            # Clone the repo
            subprocess.run(['git', 'clone', self.repo_url], 
                          check=True, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            
            logger.info(f"Repository cloned successfully to {self.repo_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during repository cloning: {e}")
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
        rel_path = os.path.relpath(file_path, start=self.repo_name)
        
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
        rel_path = os.path.relpath(file_path, start=self.repo_name)
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
    
    def run(self) -> bool:
        """
        Run the complete documentation process.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clone the repository
            if not self.clone_repository():
                return False
            
            # Find files to document
            files = self.find_files_to_document()
            
            if not files:
                logger.warning("No files found to document")
                return False
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Generate documentation for each file
            doc_files = []
            for i, file_path in enumerate(files):
                logger.info(f"Processing file {i+1}/{len(files)}: {file_path}")
                
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
            
            # Generate index
            self.generate_index(doc_files)
            
            logger.info(f"Documentation completed successfully. Output in {self.output_dir} directory")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error during documentation process: {e}")
            return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate documentation for a repository using GPT-4'
    )
    
    parser.add_argument('repo_url', help='URL of the repository to document')
    parser.add_argument('--output-dir', '-o', default='docs', 
                        help='Directory where documentation will be written')
    parser.add_argument('--api-key', help='OpenAI API key (defaults to OPENAI_API_KEY env variable)')
    parser.add_argument('--model', default='gpt-4', 
                        help='OpenAI model to use (default: gpt-4)')
    parser.add_argument('--file-types', nargs='+', 
                        help=f'File extensions to document (default: {DEFAULT_FILE_EXTENSIONS})')
    parser.add_argument('--ignore', nargs='+', 
                        help='Additional glob patterns to ignore')
    parser.add_argument('--max-tokens', type=int, default=4000, 
                        help='Maximum tokens for API responses (default: 4000)')
    parser.add_argument('--temperature', type=float, default=0.2, 
                        help='Temperature setting for GPT-4 responses (default: 0.2)')
    parser.add_argument('--max-files', type=int, 
                        help='Maximum number of files to process')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Process file extensions
    file_extensions = args.file_types
    if file_extensions:
        # Ensure all extensions start with a dot
        file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions]
    
    # Combine default and user-specified ignore patterns
    ignore_patterns = DEFAULT_IGNORE_PATTERNS
    if args.ignore:
        ignore_patterns.extend(args.ignore)
    
    # Create and run the documentor
    documentor = CodeDocumentor(
        repo_url=args.repo_url,
        output_dir=args.output_dir,
        api_key=args.api_key,
        model=args.model,
        file_extensions=file_extensions,
        ignore_patterns=ignore_patterns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_files=args.max_files
    )
    
    success = documentor.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
