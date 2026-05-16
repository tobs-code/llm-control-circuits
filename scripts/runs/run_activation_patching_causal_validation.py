#!/usr/bin/env python3
"""
Activation Patching Framework for Causal Validation

This script implements activation patching and interchange interventions to:
1. Test which components cause observed collapse modes
2. Separate shared decoder effects from direction-specific effects  
3. Provide mechanistic validation for "separate directions, shared decoder" hypothesis

Key experiments:
- Direction patching: Patch individual steering directions
- Component patching: Patch specific decoder components (MLP, Attention, LN)
- Interchange interventions: Swap activations between clean/corrupted runs
- Ablation patching: Systematically remove/restore components
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PatchingConfig:
    """Configuration for activation patching experiments"""
    model_name: str
    clean_prompt: str
    corrupted_prompt: str
    target_layers: List[int]
    hook_targets: List[str]  # ['block', 'mlp', 'attn', 'ln1', 'ln2']
    patch_directions: List[str]  # ['refusal', 'propaganda', 'concept']
    alphas: Dict[str, float]
    output_dir: str
    
class ActivationStore:
    """Store activations from clean and corrupted runs"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.activations = {}
        self.hooks = []
        
    def add_hook(self, hook_name, target_layers, hook_targets):
        """Add hooks to capture activations"""
        def make_hook(layer_idx, target):
            def hook_fn(module, input, output):
                key = f"{target}_{layer_idx}"
                if isinstance(output, tuple):
                    self.activations[key] = output[0].detach().clone()
                else:
                    self.activations[key] = output.detach().clone()
            return hook_fn
            
        for layer_idx in target_layers:
            if target == 'block':
                layer = self.model.model.layers[layer_idx]
            elif target == 'mlp':
                layer = self.model.model.layers[layer_idx].mlp
            elif target == 'attn':
                layer = self.model.model.layers[layer_idx].self_attn
            elif target == 'ln1':
                layer = self.model.model.layers[layer_idx].input_layernorm
            elif target == 'ln2':
                layer = self.model.model.layers[layer_idx].post_attention_layernorm
            else:
                continue
                
            hook = layer.register_forward_hook(make_hook(layer_idx, target))
            self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def capture_activations(self, prompt: str, steering_vectors: Dict = None):
        """Capture activations for a given prompt"""
        self.clear_hooks()
        self.activations = {}
        
        # Add hooks for this capture
        self.add_hook("capture", list(range(self.model.config.num_hidden_layers)), 
                     ['block', 'mlp', 'attn', 'ln1', 'ln2'])
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Apply steering if provided
        if steering_vectors:
            # TODO: Implement steering application
            pass
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Store captured activations
        captured = {k: v.clone() for k, v in self.activations.items()}
        self.clear_hooks()
        return captured

class ActivationPatcher:
    """Apply activation patches for causal validation"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.patch_hooks = []
        
    def apply_patch(self, patch_activations: Dict, patch_config: Dict):
        """Apply patches to specific components"""
        def make_patch_hook(layer_idx, target, patch_tensor):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    # Replace first element (usually hidden states)
                    patched = patch_tensor.to(output[0].device)
                    return (patched,) + output[1:]
                else:
                    return patch_tensor.to(output.device)
            return hook_fn
            
        # Clear existing patches
        self.clear_patches()
        
        # Apply new patches
        for key, patch_tensor in patch_activations.items():
            if '_' not in key:
                continue
                
            target, layer_str = key.rsplit('_', 1)
            try:
                layer_idx = int(layer_str)
            except ValueError:
                continue
                
            # Check if this target should be patched
            if target in patch_config.get('hook_targets', []):
                if layer_idx in patch_config.get('target_layers', []):
                    
                    # Get the correct module
                    if target == 'block':
                        module = self.model.model.layers[layer_idx]
                    elif target == 'mlp':
                        module = self.model.model.layers[layer_idx].mlp
                    elif target == 'attn':
                        module = self.model.model.layers[layer_idx].self_attn
                    elif target == 'ln1':
                        module = self.model.model.layers[layer_idx].input_layernorm
                    elif target == 'ln2':
                        module = self.model.model.layers[layer_idx].post_attention_layernorm
                    else:
                        continue
                        
                    hook = module.register_forward_hook(make_patch_hook(layer_idx, target, patch_tensor))
                    self.patch_hooks.append(hook)
                    logger.info(f"Applied patch: {target} layer {layer_idx}")
    
    def clear_patches(self):
        """Remove all patch hooks"""
        for hook in self.patch_hooks:
            hook.remove()
        self.patch_hooks = []

class CausalValidationRunner:
    """Main runner for causal validation experiments"""
    
    def __init__(self, config: PatchingConfig):
        self.config = config
        self.setup_model()
        self.activation_store = ActivationStore(self.model, self.tokenizer)
        self.patcher = ActivationPatcher(self.model, self.tokenizer)
        
    def setup_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def run_direction_patching(self):
        """Experiment 1: Patch individual steering directions"""
        logger.info("=== Direction Patching Experiment ===")
        
        # Capture clean activations (no steering)
        clean_acts = self.activation_store.capture_activations(self.config.clean_prompt)
        
        # Capture corrupted activations (with steering)
        # TODO: Apply steering vectors
        corrupted_acts = self.activation_store.capture_activations(self.config.corrupted_prompt)
        
        results = []
        
        # Patch each direction separately
        for direction in self.config.patch_directions:
            logger.info(f"Testing direction: {direction}")
            
            # Create patch dictionary for this direction
            patch_dict = {}
            for key in clean_acts.keys():
                if direction in key.lower():  # Simple heuristic
                    patch_dict[key] = corrupted_acts[key]
                    
            # Apply patch and test
            patch_config = {
                'hook_targets': self.config.hook_targets,
                'target_layers': self.config.target_layers
            }
            
            self.patcher.apply_patch(patch_dict, patch_config)
            
            # Generate output
            inputs = self.tokenizer(self.config.clean_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                'experiment': 'direction_patching',
                'direction': direction,
                'prompt': self.config.clean_prompt,
                'output': generated_text,
                'patched_components': list(patch_dict.keys())
            }
            results.append(result)
            
            self.patcher.clear_patches()
            
        return results
    
    def run_component_patching(self):
        """Experiment 2: Patch specific decoder components"""
        logger.info("=== Component Patching Experiment ===")
        
        # Capture baseline activations
        clean_acts = self.activation_store.capture_activations(self.config.clean_prompt)
        corrupted_acts = self.activation_store.capture_activations(self.config.corrupted_prompt)
        
        results = []
        
        # Test each component type
        for component in self.config.hook_targets:
            logger.info(f"Testing component: {component}")
            
            # Create patch dictionary for this component
            patch_dict = {}
            for key in corrupted_acts.keys():
                if key.startswith(component):
                    patch_dict[key] = corrupted_acts[key]
                    
            # Apply patch
            patch_config = {
                'hook_targets': [component],
                'target_layers': self.config.target_layers
            }
            
            self.patcher.apply_patch(patch_dict, patch_config)
            
            # Generate output
            inputs = self.tokenizer(self.config.clean_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                'experiment': 'component_patching',
                'component': component,
                'prompt': self.config.clean_prompt,
                'output': generated_text,
                'patched_components': list(patch_dict.keys())
            }
            results.append(result)
            
            self.patcher.clear_patches()
            
        return results
    
    def run_interchange_intervention(self):
        """Experiment 3: Interchange interventions between runs"""
        logger.info("=== Interchange Intervention Experiment ===")
        
        # Capture activations from different intervention types
        baseline_acts = self.activation_store.capture_activations(self.config.clean_prompt)
        
        # TODO: Capture from different steering configurations
        bypass_acts = self.activation_store.capture_activations(self.config.corrupted_prompt)
        concept_acts = self.activation_store.capture_activations(self.config.clean_prompt)  # With concept steering
        
        results = []
        
        # Test different interchange combinations
        interventions = [
            ('bypass_to_concept', bypass_acts, concept_acts),
            ('concept_to_bypass', concept_acts, bypass_acts),
            ('bypass_to_baseline', bypass_acts, baseline_acts),
            ('concept_to_baseline', concept_acts, baseline_acts),
        ]
        
        for intervention_name, source_acts, target_acts in interventions:
            logger.info(f"Testing interchange: {intervention_name}")
            
            # Apply interchange patches
            patch_config = {
                'hook_targets': self.config.hook_targets,
                'target_layers': self.config.target_layers
            }
            
            self.patcher.apply_patch(source_acts, patch_config)
            
            # Generate output
            inputs = self.tokenizer(self.config.clean_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                'experiment': 'interchange_intervention',
                'intervention': intervention_name,
                'prompt': self.config.clean_prompt,
                'output': generated_text,
                'source_components': list(source_acts.keys()),
                'target_components': list(target_acts.keys())
            }
            results.append(result)
            
            self.patcher.clear_patches()
            
        return results
    
    def run_ablation_patching(self):
        """Experiment 4: Systematic ablation and restoration"""
        logger.info("=== Ablation Patching Experiment ===")
        
        # Capture corrupted activations
        corrupted_acts = self.activation_store.capture_activations(self.config.corrupted_prompt)
        
        results = []
        
        # Test progressive restoration
        for layer_idx in self.config.target_layers:
            logger.info(f"Testing layer restoration: {layer_idx}")
            
            # Restore only this layer
            patch_dict = {}
            for key in corrupted_acts.keys():
                if f"_{layer_idx}" in key:
                    patch_dict[key] = corrupted_acts[key]
                    
            # Apply partial restoration
            patch_config = {
                'hook_targets': self.config.hook_targets,
                'target_layers': [layer_idx]
            }
            
            self.patcher.apply_patch(patch_dict, patch_config)
            
            # Generate output
            inputs = self.tokenizer(self.config.clean_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                'experiment': 'ablation_patching',
                'restored_layer': layer_idx,
                'prompt': self.config.clean_prompt,
                'output': generated_text,
                'restored_components': list(patch_dict.keys())
            }
            results.append(result)
            
            self.patcher.clear_patches()
            
        return results
    
    def run_all_experiments(self):
        """Run all causal validation experiments"""
        logger.info("Starting Causal Validation Experiments")
        
        all_results = []
        
        # Run all experiment types
        all_results.extend(self.run_direction_patching())
        all_results.extend(self.run_component_patching())
        all_results.extend(self.run_interchange_intervention())
        all_results.extend(self.run_ablation_patching())
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.output_dir) / f"causal_validation_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': self.config.__dict__,
                'results': all_results,
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to: {output_file}")
        
        # Generate markdown summary
        self.generate_markdown_summary(all_results, output_file.with_suffix('.md'))
        
        return all_results
    
    def generate_markdown_summary(self, results: List[Dict], output_file: Path):
        """Generate markdown summary of results"""
        summary = f"""# Causal Validation Results
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {self.config.model_name}

## Experiments Run

"""
        
        # Group results by experiment type
        by_experiment = {}
        for result in results:
            exp_type = result['experiment']
            if exp_type not in by_experiment:
                by_experiment[exp_type] = []
            by_experiment[exp_type].append(result)
        
        for exp_type, exp_results in by_experiment.items():
            summary += f"### {exp_type.replace('_', ' ').title()}\n\n"
            
            for i, result in enumerate(exp_results[:3]):  # Show first 3 results
                summary += f"#### Result {i+1}\n"
                if 'direction' in result:
                    summary += f"Direction: `{result['direction']}`\n"
                if 'component' in result:
                    summary += f"Component: `{result['component']}`\n"
                if 'intervention' in result:
                    summary += f"Intervention: `{result['intervention']}`\n"
                if 'restored_layer' in result:
                    summary += f"Restored Layer: `{result['restored_layer']}`\n"
                    
                summary += f"Prompt: `{result['prompt']}`\n\n"
                summary += "Output:\n```\n"
                summary += result['output'][:500] + ("..." if len(result['output']) > 500 else "")
                summary += "\n```\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        logger.info(f"Markdown summary saved to: {output_file}")

def main():
    """Main function to run causal validation"""
    
    # Example configuration - adapt as needed
    config = PatchingConfig(
        model_name="Qwen/Qwen2.5-1.5B",
        clean_prompt="Tell me about Tiananmen 1989.",
        corrupted_prompt="Tell me about Tiananmen 1989.",  # Will be modified with steering
        target_layers=[20, 21, 22, 23],  # Late layers where control happens
        hook_targets=['block', 'mlp', 'attn'],
        patch_directions=['refusal', 'propaganda', 'concept'],
        alphas={'refusal': 1.6, 'propaganda': 2.2, 'concept': 6.0},
        output_dir="results"
    )
    
    runner = CausalValidationRunner(config)
    results = runner.run_all_experiments()
    
    logger.info(f"Causal validation complete. {len(results)} experiments run.")

if __name__ == "__main__":
    main()
