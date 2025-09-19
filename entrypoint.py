import requests
import openai
import click
import os
import base64
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
import google.generativeai as genai

load_dotenv()

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Set API keys via environment variables for security
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_KEY = os.getenv("API_KEY")
os.environ["OLLAMA_HOST"] = "http://localhost:11434"


# GitHub API Headers (only if authentication is needed)
AUTH_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"} if GITHUB_TOKEN else {}

def is_repo_public(repo_owner, repo_name):
    """Check if a repository is public."""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    response = requests.get(url)  # Unauthenticated request
    if response.status_code == 200:
        return True  # Public repo
    elif response.status_code == 404:
        click.echo(f"Error: Repository {repo_owner}/{repo_name} not found.")
    else:
        click.echo(f"Error: Unable to determine repository visibility (Status: {response.status_code}, Response: {response.text}).")
    return False

def fetch_pr_files(repo_owner, repo_name, pr_number, public_repo):
    """Fetch the modified files and their diffs in the pull request."""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/files"
    headers = {} if public_repo else AUTH_HEADERS
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        click.echo(f"Error fetching PR files (Status: {response.status_code}, Response: {response.text}).")
        return []
    return response.json()

def fetch_file_content(repo_owner, repo_name, file_path, branch, public_repo):
    """Fetch the full content of a file from the repository."""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}?ref={branch}"
    headers = {} if public_repo else AUTH_HEADERS
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        click.echo(f"Error fetching file {file_path} (Status: {response.status_code}, Response: {response.text}).")
        return ""
    file_data = response.json()
    if 'content' in file_data:
        return base64.b64decode(file_data['content']).decode('utf-8')
    return ""

def review_code(diff, filename, full_code, model):
    """Generate a code review using OpenAI or Ollama based on the chosen model."""
    provider, model_name = model.split(":", 1)
    
    prompt = f"""
    You are a code reviewer. Review the following Git diff from {filename} for potential bugs.
    Use the full file content as context, but focus only on the changes.
    Suggest improvements where necessary. Keep it concise, don't highlight positives, just negatives.
    If there are more than 3 suggestions for a file, pick the top suggestion only, unless there is a critical one.
    Show before and after code snippets where possible. Be encouraging.
    
    Full file content:
    ```python
    {full_code}
    ```
    
    Changes:
    ```diff
    {diff}
    ```
    
    Provide constructive feedback with clear recommendations.
    """
    
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=API_KEY)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        elif provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise RuntimeError("Ollama module is not installed. Install it using 'pip install ollama'.")
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}],host="http://localhost:11434")
            return response["message"]["content"]

        elif provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise RuntimeError("Google Gemini module not installed. Install with 'pip install google-generativeai'.")
            if not API_KEY:
                raise RuntimeError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=API_KEY)
            model_obj = genai.GenerativeModel(model_name)
            response = model_obj.generate_content(prompt)
            # response.text contains the full generated review
            return response.text
        
        
        else:
            raise ValueError("Invalid model provider. Use 'openai' or 'ollama'.")
    except Exception as e:
        click.echo(f"Error during code review: {e}")
        return ""
def summarize_and_email(
    review_file_path: str,
    recipient_email: str,
    subject: str = "Code Review Summary",
):
    """
    Read a text file containing code review details,
    create a summary using Gemini, and send it as an email.
    """
    # --- Load file ---
    if not os.path.exists(review_file_path):
        raise FileNotFoundError(f"{review_file_path} does not exist")
    with open(review_file_path, "r", encoding="utf-8") as f:
        review_text = f.read()

    # --- Summarize with Gemini (or replace with OpenAI etc.) ---
    gemini_key = os.getenv("API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment variables.")
    genai.configure(api_key=gemini_key)

    prompt = f"""
    Create a concise email-ready summary of the following code review report.
    Highlight the most important findings, potential bugs, and recommendations.
    Be clear and professional.

    Review file content:
    {review_text}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    summary = model.generate_content(prompt).text

    # --- Prepare email ---
    sender_email = os.getenv("SENDER_EMAIL")        # must be set as a secret
    sender_password = os.getenv("SENDER_PASSWORD")  # app password for Gmail/SMTP
    if not sender_email or not sender_password:
        raise RuntimeError("Set SENDER_EMAIL and SENDER_PASSWORD environment variables.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email
    message.set_content(summary)

    # --- Send email ---
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(message)

    print(f"✅ Summary emailed to {recipient_email}")


@click.command()
@click.option("--repo",required=True,help="repository name")
@click.option("--pr_number", type=int,required=True,help="pr-number")
@click.option("--model", required=True, help="model name is required")
@click.option("--pr_authoremail",required=True,help="email")
def cli(repo, pr_number, model,pr_authoremail):
    """Fetch and review a GitHub PR using OpenAI or Ollama with full file context."""
    print(repo,pr_number,model,"email:",pr_authoremail)
    repo_owner, repo_name = repo.split("/")
    public_repo = is_repo_public(repo_owner, repo_name)
    
    if not public_repo and not GITHUB_TOKEN:
        click.echo("Error: Private repository detected. Please set GITHUB_TOKEN as an environment variable.")
        return
    
    provider, _ = model.split(":", 1)
    if provider == "openai" and not OPENAI_API_KEY:
        click.echo("Error: Please set OPENAI_API_KEY as an environment variable.")
        return
    output_file = "/github/workspace/review_results.txt"
    # Fetch PR details to get the branch name
    pr_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"
    headers = {} if public_repo else AUTH_HEADERS
    pr_response = requests.get(pr_url, headers=headers)
    click.echo(pr_response)
    if pr_response.status_code != 200:
        click.echo(f"Error fetching PR details (Status: {pr_response.status_code}, Response: {pr_response.text}).")
        return
    pr_data = pr_response.json()
    branch = pr_data['head']['ref']
    
    files = fetch_pr_files(repo_owner, repo_name, pr_number, public_repo)
    if not files:
        click.echo("No files retrieved from the PR. Check repository and PR number.")
        return
    click.echo(len(files))
    for file in files:
        filename = file['filename']
        
        diff = file.get('patch', '')
        
        if not diff:
            click.echo(f"No changes detected in {filename}")
            continue
        
        full_code = fetch_file_content(repo_owner, repo_name, filename, branch, public_repo)
        click.echo(f"\nReviewing changes in: {filename}")
        review = review_code(diff, filename, full_code, model)
        click.echo(f"\nReview for {filename}:{review}\n{'-'*40}")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n### Review for {filename}\n{review}\n{'-'*40}\n")
    click.echo("sending mail")
    try:
        summarize_and_email(output_file,pr_authoremail)
    except Exception as e:
        click.echo(e)

if __name__ == "__main__":
    cli()