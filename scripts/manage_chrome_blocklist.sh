#!/bin/bash

BLOCKLIST_FILE="$HOME/amitejmehta/scripts/chrome-policy-block.plist"
TARGET_POLICY="/Library/Managed Preferences/com.google.Chrome.plist"

# Initialize blocklist if missing
init_blocklist() {
  if [ ! -f "$BLOCKLIST_FILE" ]; then
    echo "Creating new blocklist..."
    cat <<EOF > "$BLOCKLIST_FILE"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>URLBlocklist</key>
  <array>
  </array>
</dict>
</plist>
EOF
  fi
}

# Show current blocked domains
show_blocklist() {
  echo "Current blocked domains:"
  grep -oE '<string>([^<]*)</string>' "$BLOCKLIST_FILE" | sed -E 's/<string>(https:\/\/www\.|https:\/\/)?//' | sed -E 's/<\/string>//' | sed -E 's/^www\.//' | sed 's/\/$//' | sort -u
}

# Add a URL to the blocklist
add_url() {
  DOMAIN="$1"

  # Check if already exists
  if grep -q "$DOMAIN" "$BLOCKLIST_FILE"; then
    echo "$DOMAIN already exists in blocklist."
    return
  fi

  # Safely insert new domain entries inside the <array>
  awk -v domain="$DOMAIN" '
    /<array>/ { print; inside=1; next }
    /<\/array>/ {
      if (inside) {
        print "    <string>https://www."domain"</string>"
        print "    <string>"domain"</string>"
        inside=0
      }
      print; next
    }
    { print }
  ' "$BLOCKLIST_FILE" > "$BLOCKLIST_FILE.tmp" && mv "$BLOCKLIST_FILE.tmp" "$BLOCKLIST_FILE"

  echo "$DOMAIN added to blocklist."
}

# Remove a URL from blocklist
remove_url() {
  DOMAIN="$1"
  sed -i '' "/https:\/\/www\.$DOMAIN/d" "$BLOCKLIST_FILE"
  sed -i '' "/<string>$DOMAIN<\/string>/d" "$BLOCKLIST_FILE"
  echo "$DOMAIN removed from blocklist."
}

# Update the policy and restart Chrome
update_policy() {
  echo "Validating blocklist..."
  plutil "$BLOCKLIST_FILE"
  if [ $? -ne 0 ]; then
    echo "Blocklist is invalid. Please fix it manually."
    exit 1
  fi

  echo "Copying blocklist to managed preferences..."
  sudo cp "$BLOCKLIST_FILE" "$TARGET_POLICY"

  echo "Refreshing macOS preferences..."
  sudo killall cfprefsd

  echo "Restarting Chrome..."
  osascript -e 'quit app "Google Chrome"'
  sleep 2
  open -a "Google Chrome"

  echo "✅ Blocklist updated and Chrome restarted."
}

# Menu
init_blocklist

echo "Quitting Tailscale to ensure clean environment..."
osascript -e 'quit app "Tailscale"'
sleep 2

while true; do
  echo ""
  show_blocklist
  echo ""
  echo "Choose an option:"
  echo "1) Add a domain to blocklist"
  echo "2) Remove a domain from blocklist"
  echo "3) Update Chrome policy and reload"
  echo "4) Exit"
  read -rp "Enter choice [1-4]: " choice

  case "$choice" in
    1)
      read -rp "Enter domain to block (e.g., youtube.com): " domain
      add_url "$domain"
      ;;
    2)
      read -rp "Enter domain to remove (e.g., youtube.com): " domain
      remove_url "$domain"
      ;;
    3)
      update_policy
      ;;
    4)
      echo "Restarting Tailscale..."
      open -a "Tailscale"
      sleep 2
      exit 0
      ;;
    *)
      echo "Invalid option."
      ;;
  esac
done

