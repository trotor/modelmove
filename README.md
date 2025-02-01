# ModelMove

ModelMove is a command-line tool for managing Ollama model storage. It allows you to move large language models between your main storage and an offload location, helping you manage disk space while keeping your models accessible.

## Features

- Move models from Ollama storage to offload storage
- Restore models back to Ollama when needed
- List models in both storages with sizes
- Verbose mode for detailed operation information
- Configurable storage locations via environment variables

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/modelmove.git
cd modelmove
```

2. Install using Poetry:
```bash
poetry install
```

## Configuration

Create a `.env` file in your home directory or project directory:
```bash
# Required: Path to offload storage
OFFLOAD_PATH=~/model_storage

# Optional: Override Ollama models path (defaults to ~/.ollama/models)
OLLAMA_PATH=~/.ollama/models
```

## Usage

Basic commands:
```bash
# List all models in both storages
modelmove --list

# Move a model to offload storage
modelmove -o llama2:13b

# Restore a model to Ollama storage
modelmove -r llama2:13b

# Show detailed information
modelmove -v --list
```

Full command reference:
```
usage: modelmove [-h] [--main-storage MAIN_STORAGE] [--offload-storage OFFLOAD_STORAGE] [--list] 
                 [--move-to-offload MODEL] [--move-to-main MODEL] [--version] [--verbose]

Model storage management tool

options:
  -h, --help            show this help message and exit
  --main-storage MAIN_STORAGE
                        Main storage path (default: ~/.ollama/models)
  --offload-storage OFFLOAD_STORAGE
                        Offload storage path (default: ~/model_storage)
  --list, -l           List models in both storages (default action)
  --move-to-offload MODEL, --offload MODEL, -o MODEL
                        Move model to offload storage
  --move-to-main MODEL, --restore MODEL, -r MODEL
                        Move model back to main storage
  --version            Show program version
  --verbose, -v        Show detailed information
```

## Example Workflow

1. Check available models:
```bash
modelmove --list
```

2. Move a large model to offload storage:
```bash
modelmove -o llama2:70b --verbose
```

3. When needed, restore the model:
```bash
modelmove -r llama2:70b
```

## Requirements

- Python 3.8 or newer
- Ollama installed
- Sufficient disk space in offload location

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
modelmove -r llama2:70b
```
