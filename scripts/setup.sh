# Ultra-simple script to update .zshrc functions
echo "Setting up Ami's utility functions in .zshrc..."
ZSHRC="$HOME/.zshrc"

# Safe backup
cp "$ZSHRC" "$ZSHRC.bak"

# Define markers and functions
START="# === BEGIN AMI UTILITY FUNCTIONS ==="
END="# === END AMI UTILITY FUNCTIONS ==="

# Create the function block (single quotes to prevent variable expansion)
BLOCK='# === BEGIN AMI UTILITY FUNCTIONS ===
login() {
    if [ -z "$1" ]; then
        echo "Usage: login <profile_name> (dev|prod)"
        return 1
    fi
    export AWS_PROFILE=$1 && aws sso login --profile $1
}

repo() {
    if [ -z "$1" ]; then
        echo "Usage: repo <repo_name>"
        return 1
    fi
    cd ~/$1
}

alias blocklist="~/amitejmehta/scripts/manage_chrome_blocklist.sh"
alias activate="source .venv/bin/activate"
# === END AMI UTILITY FUNCTIONS ==='

# Check if the functions are already in the file
if grep -q "$START" "$ZSHRC"; then
    # Delete everything between and including markers
    sed -i.bak "/$START/,/$END/d" "$ZSHRC"
    echo "Removed old functions block"
fi

# Append the block to the end of the file
echo "$BLOCK" >> "$ZSHRC"
echo "✓ Added functions to .zshrc"
echo "Please run 'source ~/.zshrc' to use the updated functions"