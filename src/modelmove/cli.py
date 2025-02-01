import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import json
from dataclasses import dataclass
import subprocess
import sys

VERSION = "0.1.0"
DEFAULT_OLLAMA_PATH = Path.home() / ".ollama" / "models"

@dataclass
class OllamaModel:
    name: str
    manifest_path: Path
    manifest_data: Dict[str, Any]
    
    @property
    def model_size(self) -> float:
        """Get total size of model layers in MB"""
        total_size = sum(layer['size'] for layer in self.manifest_data.get('layers', []))
        return total_size / (1024 * 1024)
    
    @property
    def version(self) -> str:
        """Get model version from manifest path"""
        return self.manifest_path.name
    
    @property
    def full_name(self) -> str:
        """Get full model name with version"""
        model_name = self.manifest_path.parent.name
        return f"{model_name}:{self.version}"
    
    @property
    def required_files(self) -> List[Dict[str, str]]:
        """Get list of all required files from manifest"""
        files = []
        # Add config file if exists
        if 'config' in self.manifest_data:
            files.append({
                'digest': self.manifest_data['config']['digest'],
                'size': self.manifest_data['config']['size'],
                'type': 'config'
            })
        # Add all layer files
        for layer in self.manifest_data.get('layers', []):
            files.append({
                'digest': layer['digest'],
                'size': layer['size'],
                'type': layer['mediaType']
            })
        return files

class ModelManager:
    def __init__(self, main_storage: str, offload_storage: str, verbose: bool = False):
        self.main_storage = Path(main_storage)
        self.offload_storage = Path(offload_storage)
        self.verbose = verbose
        
        # Define Ollama specific paths
        self.ollama_models_path = self.main_storage / "manifests" / "registry.ollama.ai" / "library"
        self.ollama_blobs_path = self.main_storage / "blobs"
        
        if self.verbose:
            print(f"Blobs path: {self.ollama_blobs_path}")
            if self.ollama_blobs_path.exists():
                print("Blobs directory exists")
                # List some blobs
                for blob in list(self.ollama_blobs_path.iterdir())[:5]:
                    print(f"Found blob: {blob}")
        
        # Ensure directories exist
        self.main_storage.mkdir(parents=True, exist_ok=True)
        self.offload_storage.mkdir(parents=True, exist_ok=True)

    def get_model_details(self, model_dir: Path) -> OllamaModel:
        """Read model manifest and return details"""
        if self.verbose:
            print(f"Checking model directory: {model_dir}")
        
        # Look for version files directly in version directories
        for version_dir in model_dir.iterdir():
            if version_dir.is_file() and not version_dir.name.startswith('.'):
                if self.verbose:
                    print(f"  Found version file: {version_dir}")
                
                try:
                    with open(version_dir, 'r') as f:
                        manifest_data = json.load(f)
                        model = OllamaModel(
                            name=model_dir.name,  # e.g., "llama3.2"
                            manifest_path=version_dir,  # e.g., "path/to/llama3.2/3b"
                            manifest_data=manifest_data
                        )
                        if self.verbose:
                            print(f"  Successfully loaded model: {model.full_name}")
                        return model
                except json.JSONDecodeError:
                    if self.verbose:
                        print(f"  Warning: Failed to parse manifest in {version_dir}")
                    continue
                except Exception as e:
                    if self.verbose:
                        print(f"  Error reading {version_dir}: {e}")
        return None

    def list_models(self) -> tuple[List[OllamaModel], List[str]]:
        """List models in both storages."""
        # For Ollama storage, look in the manifests/registry.ollama.ai/library directory
        main_models = []
        if self.ollama_models_path.exists():
            for model_dir in self.ollama_models_path.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    model = self.get_model_details(model_dir)
                    if model:
                        main_models.append(model)
        
        # For offload storage, list model directories
        offload_models = []
        if self.offload_storage.exists():
            for model_dir in self.offload_storage.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    offload_models.append(model_dir.name)
        
        return main_models, offload_models

    def _parse_model_name(self, full_name: str) -> tuple[str, str]:
        """Parse model name and version from full name (e.g., 'llama3.2:3b' -> ('llama3.2', '3b'))"""
        if ':' in full_name:
            model_name, version = full_name.split(':', 1)
            return model_name, version
        return full_name, None

    def _stop_ollama_model(self, model_name: str) -> None:
        """Stop running Ollama model"""
        if self.verbose:
            print(f"Stopping Ollama model: {model_name}")
        try:
            subprocess.run(['ollama', 'kill'], check=True, capture_output=True, text=True)
            if self.verbose:
                print("Ollama model stopped successfully")
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"Warning: Failed to stop Ollama model: {e.stderr}")
        except FileNotFoundError:
            if self.verbose:
                print("Warning: Ollama command not found")

    def move_to_offload(self, model_name: str) -> None:
        """Move model from main storage to offload storage."""
        model_name, version = self._parse_model_name(model_name)
        
        # Stop Ollama model before moving
        self._stop_ollama_model(model_name)
        
        source_dir = self.ollama_models_path / model_name
        
        if not source_dir.exists():
            # Get list of available models for better error message
            available_models = []
            if self.ollama_models_path.exists():
                for model_dir in self.ollama_models_path.iterdir():
                    if model_dir.is_dir() and not model_dir.name.startswith('.'):
                        model = self.get_model_details(model_dir)
                        if model:
                            available_models.append(model.full_name)
            
            error_msg = [f"Model '{model_name}' not found in Ollama storage"]
            if available_models:
                error_msg.append("\nAvailable models:")
                for model in sorted(available_models):
                    error_msg.append(f"  - {model}")
            else:
                error_msg.append("\nNo models found in Ollama storage")
            
            raise FileNotFoundError('\n'.join(error_msg))
        
        # If version is specified, verify it exists and get model details
        manifest_file = source_dir / version
        if not manifest_file.exists():
            raise FileNotFoundError(f"Version '{version}' not found for model '{model_name}'")
        
        # Get model details to find all required files
        model = self.get_model_details(source_dir)
        if not model:
            raise FileNotFoundError(f"Failed to read model details for '{model_name}'")
        
        # Create model directory in offload storage
        model_offload_dir = self.offload_storage / f"{model_name}-{version}"
        model_offload_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\nMoving files to: {model_offload_dir}")
            print(f"Found {len(model.required_files)} required files")
        
        # Move manifest file
        if self.verbose:
            print(f"\nMoving manifest file:")
            print(f"  From: {manifest_file}")
            print(f"  To: {model_offload_dir / version}")
        shutil.move(manifest_file, model_offload_dir / version)
        
        # Move all required files from manifest
        for file_info in model.required_files:
            digest = file_info['digest']
            blob_name = digest.replace(':', '-')
            blob_path = self.ollama_blobs_path / blob_name
            target_path = model_offload_dir / blob_name
            
            if self.verbose:
                print(f"\nMoving {file_info['type']}:")
                print(f"  From: {blob_path}")
                print(f"  To: {target_path}")
                print(f"  Size: {file_info['size'] / (1024*1024):.1f} MB")
            
            if blob_path.exists():
                shutil.move(str(blob_path), str(target_path))
            else:
                print(f"Warning: Required file not found: {digest}")
                print(f"  Type: {file_info['type']}")
                print(f"  Expected path: {blob_path}")
        
        # Check if the model directory is empty before removing it
        remaining_files = list(source_dir.glob('*'))
        if not remaining_files:  # Only remove if empty
            if self.verbose:
                print("\nRemoving empty model directory...")
            shutil.rmtree(source_dir)
        elif self.verbose:
            print(f"\nKeeping model directory as it contains other versions:")
            for f in remaining_files:
                print(f"  - {f.name}")

    def move_to_main(self, model_name: str) -> None:
        """Move model from offload storage to main storage."""
        model_name, version = self._parse_model_name(model_name)
        source_dir = self.offload_storage / f"{model_name}-{version}"
        
        if not source_dir.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found in offload storage")
        
        # Create necessary directories
        target_model_dir = self.ollama_models_path / model_name
        target_model_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"Restoring to: {target_model_dir}")
        
        # Move manifest file
        manifest_file = source_dir / version
        if manifest_file.exists():
            if self.verbose:
                print(f"\nMoving manifest file:")
                print(f"  From: {manifest_file}")
                print(f"  To: {target_model_dir / version}")
            shutil.move(manifest_file, target_model_dir / version)
            
            # Read manifest to get blob references
            with open(target_model_dir / version) as f:
                manifest = json.load(f)
            
            # Move all blobs back
            # First config file if exists
            if 'config' in manifest:
                digest = manifest['config']['digest']
                blob_name = digest.replace(':', '-')
                source_blob = source_dir / blob_name
                target_blob = self.ollama_blobs_path / blob_name
                
                if source_blob.exists():
                    if self.verbose:
                        print(f"\nMoving config blob:")
                        print(f"  From: {source_blob}")
                        print(f"  To: {target_blob}")
                    self.ollama_blobs_path.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_blob), str(target_blob))
                else:
                    print(f"Warning: Required blob not found: {digest}")
                    print(f"  Type: config")
                    print(f"  Expected path: {source_blob}")
            
            # Then all layer files
            for layer in manifest.get('layers', []):
                digest = layer['digest']
                blob_name = digest.replace(':', '-')
                source_blob = source_dir / blob_name
                target_blob = self.ollama_blobs_path / blob_name
                
                if source_blob.exists():
                    if self.verbose:
                        print(f"\nMoving {layer['mediaType']}:")
                        print(f"  From: {source_blob}")
                        print(f"  To: {target_blob}")
                        print(f"  Size: {layer['size'] / (1024*1024):.1f} MB")
                    self.ollama_blobs_path.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_blob), str(target_blob))
                else:
                    print(f"Warning: Required blob not found: {digest}")
                    print(f"  Type: {layer['mediaType']}")
                    print(f"  Expected path: {source_blob}")
        
        # Remove offload directory after successful restore
        if self.verbose:
            print("\nRemoving offload directory...")
        shutil.rmtree(source_dir)

def load_config():
    load_dotenv()
    
    default_main = os.getenv('MAIN_PATH', str(DEFAULT_OLLAMA_PATH))  # Käytetään oletuspolkua jos MAIN_PATH puuttuu
    default_offload = os.getenv('OFFLOAD_PATH')
    
    if not default_offload:
        print("\n❌ Configuration Error:")
        print("   OFFLOAD_PATH is not set in your .env file")
        print("\n📝 Quick Fix:")
        print("   1. Create or edit .env file in your project root")
        print("   2. Add the following line:")
        print("      OFFLOAD_PATH=/path/to/your/offload/directory")
        print("\n💡 Example:")
        print("   OFFLOAD_PATH=/Users/username/ollama_models\n")
        sys.exit(1)
        
    return default_main, default_offload

def list_storage_contents(manager: ModelManager, verbose: bool = False) -> None:
    """Display contents of both storages."""
    if verbose:
        print(f"\nScanning directories:")
        print(f"Main storage (Ollama): {manager.ollama_models_path}")
        print(f"Offload storage: {manager.offload_storage}")
    
    main_models, offload_models = manager.list_models()
    
    print("\nModels in Ollama storage:", end='')
    if not main_models:
        print(" (empty)")
    else:
        print()
        for model in sorted(main_models, key=lambda m: m.full_name):
            if verbose:
                # Show more details in verbose mode
                print(f"\n  - {model.full_name}")
                print(f"    Size: {model.model_size:.1f} MB")
                model_layer = next((layer for layer in model.manifest_data.get('layers', [])
                                  if layer['mediaType'] == 'application/vnd.ollama.image.model'), None)
                if model_layer:
                    print(f"    Model blob: {model_layer['digest']}")
            else:
                # Simple format for normal mode
                print(f"  - {model.full_name} ({model.model_size:.1f} MB)")
    
    print("\nModels in offload storage:", end='')
    if not offload_models:
        print(" (empty)")
    else:
        print()
        for model_dir in sorted(offload_models):
            model_path = manager.offload_storage / model_dir
            # Lasketaan kaikkien tiedostojen koko
            total_size = sum(f.stat().st_size for f in model_path.glob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            if verbose:
                print(f"  - {model_dir} ({size_mb:.1f} MB)")
                # Näytä myös yksittäisten tiedostojen koot
                for f in model_path.glob('*'):
                    if f.is_file():
                        file_size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"    - {f.name} ({file_size_mb:.1f} MB)")
            else:
                print(f"  - {model_dir} ({size_mb:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description='Model storage management tool')
    
    # Make path arguments optional
    parser.add_argument('--main-storage', type=str,
                      help='Main storage path (default: ~/.ollama/models)')
    parser.add_argument('--offload-storage', type=str,
                      help='Offload storage path (default: ~/model_storage)')
    parser.add_argument('--list', '-l', action='store_true',
                      help='List models in both storages (default action)')
    
    # Model movement commands with short aliases
    parser.add_argument('--move-to-offload', '--offload', '-o', type=str, metavar='MODEL',
                      help='Move model to offload storage')
    parser.add_argument('--move-to-main', '--restore', '-r', type=str, metavar='MODEL',
                      help='Move model back to main storage')
    
    parser.add_argument('--version', action='version', 
                      version=f'%(prog)s {VERSION}',
                      help='Show program version')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Show detailed information')

    args = parser.parse_args()
    
    try:
        default_main, default_offload = load_config()
        # Use command line parameters if given, otherwise use defaults
        main_storage = Path(args.main_storage) if args.main_storage else default_main
        offload_storage = Path(args.offload_storage) if args.offload_storage else default_offload
        
        manager = ModelManager(main_storage, offload_storage, args.verbose)
        
        if args.verbose:
            print(f"Configuration:")
            print(f"Main storage path: {main_storage}")
            print(f"Offload storage path: {offload_storage}")
            if manager.ollama_models_path.exists():
                print(f"Found Ollama library path: {manager.ollama_models_path}")
                print("Contents:")
                for item in manager.ollama_models_path.iterdir():
                    print(f"  - {item.name}")
        
        # If no action specified, show storage contents
        if not (args.move_to_offload or args.move_to_main) or args.list:
            list_storage_contents(manager, args.verbose)

        if args.move_to_offload:
            if args.verbose:
                print(f"\nMoving '{args.move_to_offload}' to offload storage...")
            manager.move_to_offload(args.move_to_offload)
            print(f"Moved model '{args.move_to_offload}' to offload storage")
            list_storage_contents(manager, args.verbose)

        if args.move_to_main:
            if args.verbose:
                print(f"\nMoving '{args.move_to_main}' to main storage...")
            manager.move_to_main(args.move_to_main)
            print(f"Moved model '{args.move_to_main}' to main storage")
            list_storage_contents(manager, args.verbose)
            
    except FileNotFoundError as e:
        print(f"Error: Storage directory not found - {e}")
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
    except PermissionError as e:
        print(f"Error: Permission denied accessing storage - {e}")
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        sys.exit(1)

if __name__ == '__main__':
    main() 