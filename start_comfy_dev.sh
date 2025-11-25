#!/bin/bash
# ============================================================
# ComfyUI Developer Launcher - Alex Covo Studio Edition
# ============================================================

# --- Configuration ---
COMFY_ROOT="$HOME/Documents/ComfyUI"                     # Your cloned repo path
USER_DIR="$HOME/Documents/ComfyUI/user"                  # Where your workflows & custom nodes are
INPUT_DIR="$USER_DIR/input"
OUTPUT_DIR="$USER_DIR/output"
PORT="8188"

# --- Create venv if missing ---
if [ ! -d "$COMFY_ROOT/venv" ]; then
  echo "🔧 Creating Python venv..."
  cd "$COMFY_ROOT" || exit 1
  python3 -m venv venv
  source venv/bin/activate
  echo "📦 Installing dependencies..."
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
else
  source "$COMFY_ROOT/venv/bin/activate"
fi

# --- Ensure directories exist ---
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# --- Start ComfyUI ---
echo "🚀 Launching ComfyUI..."
cd "$COMFY_ROOT" || exit 1

python main.py \
  --listen 127.0.0.1 \
  --port "$PORT" \
  --input-directory "$INPUT_DIR" \
  --output-directory "$OUTPUT_DIR" \
  --user-directory "$USER_DIR" \
  --extra-model-paths-config "$USER_DIR/extra_model_paths.yaml" &

sleep 2
open "http://127.0.0.1:$PORT"