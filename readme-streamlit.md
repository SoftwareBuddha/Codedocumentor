# Streamlit Code Documentor

A user-friendly web application for generating comprehensive documentation for code repositories using GPT-4.

## Features

- **Web-based UI**: Easy-to-use interface built with Streamlit
- **Multiple Input Methods**:
  - Document from GitHub/GitLab repositories (public or private)
  - Upload local repositories as ZIP files
- **Authentication Support**:
  - Username/password or token for private repositories
  - SSH key authentication
- **Customization Options**:
  - Select file types to document
  - Configure ignore patterns
  - Adjust GPT-4 parameters
- **Export and Download**:
  - Download generated documentation as a ZIP file
  - Preview documentation index in the app

## Requirements

- Python 3.7+
- OpenAI API key (with access to GPT-4)
- Git installed on your system
- Streamlit

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/streamlit-code-documentor.git
   cd streamlit-code-documentor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key as an environment variable (optional, can also be entered in the UI):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_code_documentor.py
   ```

2. Open your web browser to the URL shown in the console (typically http://localhost:8501)

3. Using the app:
   - Choose between documenting a GitHub repository or uploading a local repository
   - Configure authentication if using a private repository
   - Adjust advanced settings if needed
   - Click "Generate Documentation"
   - Download the generated documentation as a ZIP file

## Documentation from GitHub

For public repositories, simply enter the repository URL. For private repositories, you'll need to provide authentication:

### Username/Password or Token Authentication

1. Enter the repository URL
2. Select "Username & Password/Token" authentication method
3. Enter your username and personal access token (or password)
4. Click "Generate Documentation"

### SSH Key Authentication

1. Enter the repository URL
2. Select "SSH Key" authentication method
3. Paste your SSH private key content
4. Enter your SSH key password (if applicable)
5. Click "Generate Documentation"

## Documentation from Local Repository

1. Compress your local repository as a ZIP file
2. Select the "Upload Local Repository" tab
3. Upload the ZIP file
4. Configure settings as needed
5. Click "Generate Documentation from Upload"

## Advanced Settings

- **OpenAI API Key**: Your key for accessing GPT-4
- **Model**: Choose between GPT-4, GPT-4-turbo, or GPT-3.5-turbo
- **Maximum Files**: Limit the number of files to process
- **File Types**: Select which file extensions to document
- **Ignore Patterns**: Specify patterns for files/directories to ignore
- **Output Directory**: Directory where documentation will be generated
- **Temperature**: Control the creativity of GPT-4 responses
- **Max Tokens**: Set maximum response length for each file's documentation

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
