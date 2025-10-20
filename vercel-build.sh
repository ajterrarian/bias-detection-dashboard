#!/bin/bash
# Exit on error
set -e

# Only run if in Vercel
if [ -n "$VERCEL" ]; then
  echo "Vercel build detected - installing Node.js dependencies"
  cd dashboard
  npm install
  npm run build
  # Create a basic serverless function to handle all routes
  mkdir -p api
  echo 'module.exports = (req, res) => {
    res.redirect(301, "/dashboard/");
  }' > api/index.js
fi
