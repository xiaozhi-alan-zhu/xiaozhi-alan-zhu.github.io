#!/bin/bash
set -e

# Export common Homebrew Ruby paths (Apple Silicon and Intel)
if [ -d "/opt/homebrew/opt/ruby/bin" ]; then
  export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
fi
if [ -d "/usr/local/opt/ruby/bin" ]; then
  export PATH="/usr/local/opt/ruby/bin:$PATH"
fi

echo "Ruby version: $(ruby -v)"

echo "Installing bundler (if needed)..."
if ! gem list -i bundler >/dev/null 2>&1; then
  gem install bundler
fi

echo "Installing gems..."
bundle install

echo "Starting Jekyll local server with livereload..."
bundle exec jekyll serve --livereload --drafts
