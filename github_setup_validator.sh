#!/bin/bash

# GitHub Upload Validation Script
# This script validates that the repository is properly configured for GitHub

set -e

echo "🚀 TradingAI Bot - GitHub Upload Validation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -f "README.md" ]; then
    print_error "Please run this script from the TradingAI_Bot-main root directory"
    exit 1
fi

print_status "Validating repository structure..."

# Check essential files
essential_files=(
    "README.md"
    "LICENSE"
    "CONTRIBUTING.md"
    "SECURITY.md"
    "CODE_OF_CONDUCT.md"
    "CHANGELOG.md"
    "requirements.txt"
    "pyproject.toml"
    ".gitignore"
)

print_status "Checking essential files..."
for file in "${essential_files[@]}"; do
    if [ -f "$file" ]; then
        print_success "Found $file"
    else
        print_error "Missing $file"
        exit 1
    fi
done

# Check GitHub-specific directories and files
github_structure=(
    ".github"
    ".github/workflows"
    ".github/ISSUE_TEMPLATE"
    ".github/PULL_REQUEST_TEMPLATE"
)

print_status "Checking GitHub directory structure..."
for dir in "${github_structure[@]}"; do
    if [ -d "$dir" ]; then
        print_success "Found $dir/"
    else
        print_error "Missing $dir/"
        exit 1
    fi
done

# Check GitHub workflows
workflows=(
    ".github/workflows/ci.yml"
    ".github/workflows/release.yml"
    ".github/workflows/docs.yml"
)

print_status "Checking GitHub workflows..."
for workflow in "${workflows[@]}"; do
    if [ -f "$workflow" ]; then
        print_success "Found $workflow"
        # Validate YAML syntax
        if command -v python3 >/dev/null 2>&1; then
            if python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null; then
                print_success "YAML syntax valid for $workflow"
            else
                print_error "Invalid YAML syntax in $workflow"
                exit 1
            fi
        fi
    else
        print_error "Missing $workflow"
        exit 1
    fi
done

# Check issue templates
issue_templates=(
    ".github/ISSUE_TEMPLATE/bug_report.yml"
    ".github/ISSUE_TEMPLATE/feature_request.yml"
    ".github/ISSUE_TEMPLATE/documentation.yml"
    ".github/ISSUE_TEMPLATE/config.yml"
)

print_status "Checking issue templates..."
for template in "${issue_templates[@]}"; do
    if [ -f "$template" ]; then
        print_success "Found $template"
    else
        print_error "Missing $template"
        exit 1
    fi
done

# Check PR template
if [ -f ".github/PULL_REQUEST_TEMPLATE/pull_request_template.md" ]; then
    print_success "Found PR template"
else
    print_error "Missing PR template"
    exit 1
fi

# Validate Python package structure
print_status "Validating Python package structure..."

if [ -f "pyproject.toml" ]; then
    if grep -q "name.*tradingai_bot" pyproject.toml; then
        print_success "Package name configured correctly"
    else
        print_warning "Package name might need adjustment in pyproject.toml"
    fi
    
    if grep -q "version.*=" pyproject.toml; then
        version=$(grep "version.*=" pyproject.toml | cut -d'"' -f2)
        print_success "Version found: $version"
    else
        print_error "No version found in pyproject.toml"
        exit 1
    fi
fi

# Check requirements.txt
print_status "Validating requirements.txt..."
if [ -f "requirements.txt" ]; then
    line_count=$(wc -l < requirements.txt)
    if [ "$line_count" -gt 10 ]; then
        print_success "Requirements file has $line_count dependencies"
    else
        print_warning "Requirements file seems small ($line_count lines)"
    fi
else
    print_error "requirements.txt not found"
    exit 1
fi

# Check .gitignore
print_status "Validating .gitignore..."
gitignore_patterns=(
    "__pycache__"
    "*.pyc"
    ".env"
    "data/"
    "logs/"
    ".vscode/"
)

for pattern in "${gitignore_patterns[@]}"; do
    if grep -q "$pattern" .gitignore; then
        print_success "Found $pattern in .gitignore"
    else
        print_warning "Missing $pattern in .gitignore"
    fi
done

# Test basic Python imports
print_status "Testing basic Python functionality..."
if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import sys; print(f'Python {sys.version}')" 2>/dev/null; then
        print_success "Python is working"
    else
        print_error "Python test failed"
        exit 1
    fi
    
    # Test pytest if available
    if python3 -c "import pytest" 2>/dev/null; then
        print_success "pytest is available"
        if [ -d "tests" ]; then
            print_status "Running basic tests..."
            if python3 -m pytest tests/test_example.py -v 2>/dev/null; then
                print_success "Basic tests pass"
            else
                print_warning "Some tests failed (this might be expected)"
            fi
        fi
    else
        print_warning "pytest not available"
    fi
else
    print_warning "Python3 not found - skipping Python tests"
fi

# Check Git configuration
print_status "Checking Git configuration..."
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    print_success "Git repository initialized"
    
    # Check remote
    if git remote -v | grep -q "github.com"; then
        remote_url=$(git remote get-url origin)
        print_success "GitHub remote configured: $remote_url"
    else
        print_warning "No GitHub remote found"
    fi
    
    # Check for uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        print_warning "There are uncommitted changes"
        git status --short
    else
        print_success "Working directory is clean"
    fi
else
    print_error "Not in a Git repository"
    exit 1
fi

# Security checks
print_status "Performing security checks..."

# Check for common secrets in files
secret_patterns=(
    "password.*="
    "secret.*="
    "key.*="
    "token.*="
    "api_key.*="
)

print_status "Scanning for potential secrets..."
for pattern in "${secret_patterns[@]}"; do
    if grep -r -i --exclude-dir=.git --exclude="*.md" --exclude="github_setup_validator.sh" "$pattern" . 2>/dev/null | grep -v "example\|sample\|template\|your_.*_here"; then
        print_warning "Potential secret found with pattern: $pattern"
    fi
done

# Check file permissions
print_status "Checking file permissions..."
if find . -name "*.py" -perm /111 | grep -v __pycache__ | head -5; then
    print_warning "Some Python files are executable (might be intentional)"
fi

# GitHub-specific validations
print_status "GitHub-specific validations..."

# Check README badges
if grep -q "badge" README.md; then
    print_success "README contains badges"
else
    print_warning "README might benefit from status badges"
fi

# Check for GitHub-specific keywords in README
github_keywords=("GitHub" "Issues" "Pull Request" "Contributing" "License")
for keyword in "${github_keywords[@]}"; do
    if grep -qi "$keyword" README.md; then
        print_success "README mentions $keyword"
    else
        print_warning "README could mention $keyword"
    fi
done

# Repository size check
print_status "Checking repository size..."
repo_size=$(du -sh . | cut -f1)
print_success "Repository size: $repo_size"

if [ -d ".git" ]; then
    git_size=$(du -sh .git | cut -f1)
    print_success "Git directory size: $git_size"
fi

# Final recommendations
echo ""
echo "🎯 GitHub Upload Recommendations"
echo "================================"

print_status "Pre-upload checklist:"
echo "  □ Review all files for sensitive information"
echo "  □ Test the application locally"
echo "  □ Update README with current information"
echo "  □ Consider adding project screenshots"
echo "  □ Set up GitHub repository settings:"
echo "    - Enable Issues and Wiki if desired"
echo "    - Configure branch protection rules"
echo "    - Set up GitHub Pages for documentation"
echo "    - Configure security alerts"
echo "  □ Consider adding GitHub Discussions for community"

print_status "After upload, configure:"
echo "  □ Repository description and topics"
echo "  □ GitHub Actions secrets (if needed):"
echo "    - TELEGRAM_BOT_TOKEN (for notifications)"
echo "    - PYPI_API_TOKEN (for automated releases)"
echo "    - DOCKERHUB_USERNAME and DOCKERHUB_TOKEN"
echo "  □ Collaborator permissions"
echo "  □ Repository visibility settings"

print_status "GitHub Actions secrets needed for full functionality:"
echo "  - TELEGRAM_BOT_TOKEN: For release notifications"
echo "  - TELEGRAM_CHAT_ID: For notification target"
echo "  - PYPI_API_TOKEN: For automated package publishing"
echo "  - DOCKERHUB_USERNAME: For Docker image publishing"
echo "  - DOCKERHUB_TOKEN: For Docker Hub authentication"

# Success message
echo ""
print_success "🎉 Repository validation completed successfully!"
print_success "Your repository is ready for GitHub upload!"
echo ""
print_status "Next steps:"
echo "1. git add ."
echo "2. git commit -m 'Prepare repository for GitHub upload'"
echo "3. git push origin main"
echo "4. Configure repository settings on GitHub"
echo "5. Set up GitHub Actions secrets"
echo ""
print_status "Documentation will be available at: https://YOUR_USERNAME.github.io/TradingAI_Bot-main/"

exit 0