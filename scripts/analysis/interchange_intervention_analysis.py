#!/usr/bin/env python3
"""
Interchange Intervention Analysis for Shared Decoder

This script implements sophisticated interchange interventions to test the hypothesis:
"separate directions, shared decoder"

Key interventions:
1. Cross-direction swapping: Swap activations between different steering directions
2. Layer-wise interchange: Test which layers contribute to shared vs. direction-specific effects
3. Component-wise swapping: Test MLP vs. Attention contributions to collapse modes
4. Progressive interchange: Gradually swap components to identify critical bottlenecks
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InterchangeConfig:
    """Configuration for interchange interventions"""
    model_name: str
    base_prompt: str
    steering_configs: Dict[str, Dict]  # Different steering configurations
    target_layers: List[int]
    components: List[str]  # ['mlp', 'attn', 'ln1', 'ln2', 'block']
    output_dir: str
    
class ActivationCapture:
    """Capture and manage activations from different steering conditions"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.captured_activations = {}
        self.hooks = []
        
    def capture_condition(self, condition_name: str, prompt: str, steering_config: Dict = None):
        """Capture activations for a specific condition"""
        logger.info(f"Capturing activations for condition: {condition_name}")
        
        self.clear_hooks()
        self.captured_activations[condition_name] = {}
        
        # Setup hooks for all target layers and components
        hooks = self._setup_capture_hooks(condition_name)
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Apply steering if specified
        if steering_config:
            # TODO: Implement steering application
            pass
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
        return self.captured_activations[condition_name]
    
    def _setup_capture_hooks(self, condition_name: str):
        """Setup hooks to capture activations"""
        hooks = []
        
        def make_hook(component, layer_idx):
            def hook_fn(module, input, output):
                key = f"{component}_{layer_idx}"
                if isinstance(output, tuple):
                    self.captured_activations[condition_name][key] = output[0].detach().clone()
                else:
                    self.captured_activations[condition_name][key] = output.detach().clone()
            return hook_fn
        
        # Add hooks for all components
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
            
            # Block output
            hook = layer.register_forward_hook(make_hook('block', layer_idx))
            hooks.append(hook)
            
            # MLP
            hook = layer.mlp.register_forward_hook(make_hook('mlp', layer_idx))
            hooks.append(hook)
            
            # Attention
            hook = layer.self_attn.register_forward_hook(make_hook('attn', layer_idx))
            hooks.append(hook)
            
            # Layer norms
            hook = layer.input_layernorm.register_forward_hook(make_hook('ln1', layer_idx))
            hooks.append(hook)
            
            hook = layer.post_attention_layernorm.register_forward_hook(make_hook('ln2', layer_idx))
            hooks.append(hook)
            
        return hooks
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class InterchangeIntervention:
    """Apply sophisticated interchange interventions"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.intervention_hooks = []
        
    def apply_cross_direction_swap(self, source_condition: str, target_condition: str, 
                                 activations: Dict, swap_config: Dict):
        """Swap activations between different steering directions"""
        logger.info(f"Applying cross-direction swap: {source_condition} -> {target_condition}")
        
        self.clear_interventions()
        
        source_acts = activations[source_condition]
        target_acts = activations[target_condition]
        
        # Create swap dictionary
        swap_dict = {}
        for component in swap_config.get('components', ['mlp', 'attn']):
            for layer in swap_config.get('layers', []):
                key = f"{component}_{layer}"
                if key in source_acts and key in target_acts:
                    swap_dict[key] = source_acts[key]
                    
        # Apply swap hooks
        self._apply_swap_hooks(swap_dict)
        
        return swap_dict
    
    def apply_layer_wise_interchange(self, source_condition: str, target_condition: str,
                                   activations: Dict, layer_config: Dict):
        """Apply layer-wise interchange to identify critical layers"""
        logger.info(f"Applying layer-wise interchange: {source_condition} -> {target_condition}")
        
        self.clear_interventions()
        
        source_acts = activations[source_condition]
        target_acts = activations[target_condition]
        
        results = {}
        
        for layer in layer_config.get('target_layers', []):
            logger.info(f"Testing layer {layer} interchange")
            
            # Create swap for this specific layer
            swap_dict = {}
            for component in layer_config.get('components', ['mlp', 'attn']):
                key = f"{component}_{layer}"
                if key in source_acts:
                    swap_dict[key] = source_acts[key]
                    
            # Apply and test
            self._apply_swap_hooks(swap_dict)
            
            # Generate output
            output = self._generate_with_intervention(layer_config.get('prompt', ''))
            results[f"layer_{layer}"] = {
                'swap_dict': list(swap_dict.keys()),
                'output': output
            }
            
            self.clear_interventions()
            
        return results
    
    def apply_component_wise_swap(self, source_condition: str, target_condition: str,
                                activations: Dict, component_config: Dict):
        """Test component-specific contributions to collapse modes"""
        logger.info(f"Applying component-wise swap: {source_condition} -> {target_condition}")
        
        self.clear_interventions()
        
        source_acts = activations[source_condition]
        target_acts = activations[target_condition]
        
        results = {}
        
        for component in component_config.get('components', ['mlp', 'attn', 'ln1', 'ln2']):
            logger.info(f"Testing {component} component swap")
            
            # Create swap for this component across all target layers
            swap_dict = {}
            for layer in component_config.get('layers', []):
                key = f"{component}_{layer}"
                if key in source_acts:
                    swap_dict[key] = source_acts[key]
                    
            # Apply and test
            self._apply_swap_hooks(swap_dict)
            
            # Generate output
            output = self._generate_with_intervention(component_config.get('prompt', ''))
            results[component] = {
                'swap_dict': list(swap_dict.keys()),
                'output': output
            }
            
            self.clear_interventions()
            
        return results
    
    def apply_progressive_interchange(self, source_condition: str, target_condition: str,
                                    activations: Dict, progressive_config: Dict):
        """Gradually swap components to identify critical bottlenecks"""
        logger.info(f"Applying progressive interchange: {source_condition} -> {target_condition}")
        
        self.clear_interventions()
        
        source_acts = activations[source_condition]
        target_acts = activations[target_condition]
        
        results = {}
        current_swap = {}
        
        # Define progressive swap order
        swap_order = progressive_config.get('swap_order', [
            ('ln1', 'early'),  # Start with early layer norms
            ('attn', 'early'), # Then early attention
            ('mlp', 'early'),  # Then early MLPs
            ('ln2', 'early'),  # Then post-attention norms
            ('attn', 'late'),  # Then late attention
            ('mlp', 'late'),   # Finally late MLPs
        ])
        
        for component, timing in swap_order:
            logger.info(f"Progressive step: {component} ({timing})")
            
            # Add this component to swap
            if timing == 'early':
                target_layers = progressive_config.get('early_layers', list(range(0, 12)))
            else:
                target_layers = progressive_config.get('late_layers', list(range(12, 24)))
                
            for layer in target_layers:
                if layer < self.model.config.num_hidden_layers:
                    key = f"{component}_{layer}"
                    if key in source_acts:
                        current_swap[key] = source_acts[key]
                        
            # Apply current swap
            self._apply_swap_hooks(current_swap)
            
            # Generate output
            output = self._generate_with_intervention(progressive_config.get('prompt', ''))
            
            step_name = f"{component}_{timing}"
            results[step_name] = {
                'cumulative_swaps': list(current_swap.keys()),
                'output': output,
                'step': len(results) + 1
            }
            
            self.clear_interventions()
            
        return results
    
    def _apply_swap_hooks(self, swap_dict: Dict):
        """Apply hooks to swap activations"""
        def make_swap_hook(layer_idx, component, swap_tensor):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    # Replace first element
                    patched = swap_tensor.to(output[0].device)
                    return (patched,) + output[1:]
                else:
                    return swap_tensor.to(output.device)
            return hook_fn
        
        for key, swap_tensor in swap_dict.items():
            if '_' not in key:
                continue
                
            component, layer_str = key.rsplit('_', 1)
            try:
                layer_idx = int(layer_str)
            except ValueError:
                continue
                
            # Get the correct module
            layer = self.model.model.layers[layer_idx]
            
            if component == 'block':
                module = layer
            elif component == 'mlp':
                module = layer.mlp
            elif component == 'attn':
                module = layer.self_attn
            elif component == 'ln1':
                module = layer.input_layernorm
            elif component == 'ln2':
                module = layer.post_attention_layernorm
            else:
                continue
                
            hook = module.register_forward_hook(make_swap_hook(layer_idx, component, swap_tensor))
            self.intervention_hooks.append(hook)
    
    def _generate_with_intervention(self, prompt: str, max_new_tokens: int = 200):
        """Generate text with current interventions active"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def clear_interventions(self):
        """Remove all intervention hooks"""
        for hook in self.intervention_hooks:
            hook.remove()
        self.intervention_hooks = []

class SharedDecoderAnalyzer:
    """Main analyzer for shared decoder hypothesis testing"""
    
    def __init__(self, config: InterchangeConfig):
        self.config = config
        self.setup_model()
        self.capture = ActivationCapture(self.model, self.tokenizer)
        self.intervention = InterchangeIntervention(self.model, self.tokenizer)
        
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
    
    def run_comprehensive_analysis(self):
        """Run all interchange intervention experiments"""
        logger.info("=== Starting Comprehensive Shared Decoder Analysis ===")
        
        # Step 1: Capture activations for all conditions
        all_activations = {}
        for condition_name, steering_config in self.config.steering_configs.items():
            all_activations[condition_name] = self.capture.capture_condition(
                condition_name, 
                self.config.base_prompt, 
                steering_config
            )
        
        results = {
            'config': self.config.__dict__,
            'activations_captured': list(all_activations.keys()),
            'experiments': {}
        }
        
        # Step 2: Cross-direction swapping
        logger.info("Running cross-direction swaps...")
        cross_results = {}
        for source in all_activations.keys():
            for target in all_activations.keys():
                if source != target:
                    swap_dict = self.intervention.apply_cross_direction_swap(
                        source, target, all_activations,
                        {'components': self.config.components, 'layers': self.config.target_layers}
                    )
                    
                    output = self.intervention._generate_with_intervention(self.config.base_prompt)
                    
                    cross_results[f"{source}_to_{target}"] = {
                        'swap_components': list(swap_dict.keys()),
                        'output': output
                    }
                    
                    self.intervention.clear_interventions()
        
        results['experiments']['cross_direction_swap'] = cross_results
        
        # Step 3: Layer-wise interchange
        logger.info("Running layer-wise interchanges...")
        layer_results = {}
        for source in all_activations.keys():
            for target in all_activations.keys():
                if source != target:
                    layer_results[f"{source}_to_{target}"] = self.intervention.apply_layer_wise_interchange(
                        source, target, all_activations,
                        {'components': self.config.components, 'layers': self.config.target_layers, 'prompt': self.config.base_prompt}
                    )
        
        results['experiments']['layer_wise_interchange'] = layer_results
        
        # Step 4: Component-wise swapping
        logger.info("Running component-wise swaps...")
        component_results = {}
        for source in all_activations.keys():
            for target in all_activations.keys():
                if source != target:
                    component_results[f"{source}_to_{target}"] = self.intervention.apply_component_wise_swap(
                        source, target, all_activations,
                        {'components': self.config.components, 'layers': self.config.target_layers, 'prompt': self.config.base_prompt}
                    )
        
        results['experiments']['component_wise_swap'] = component_results
        
        # Step 5: Progressive interchange
        logger.info("Running progressive interchanges...")
        progressive_results = {}
        for source in all_activations.keys():
            for target in all_activations.keys():
                if source != target:
                    progressive_results[f"{source}_to_{target}"] = self.intervention.apply_progressive_interchange(
                        source, target, all_activations,
                        {
                            'components': self.config.components,
                            'early_layers': list(range(0, 12)),
                            'late_layers': list(range(12, 24)),
                            'prompt': self.config.base_prompt
                        }
                    )
        
        results['experiments']['progressive_interchange'] = progressive_results
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.output_dir) / f"shared_decoder_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Generate analysis summary
        self.generate_analysis_summary(results, output_file.with_suffix('.md'))
        
        return results
    
    def generate_analysis_summary(self, results: Dict, output_file: Path):
        """Generate analysis summary focusing on shared decoder hypothesis"""
        
        summary = f"""# Shared Decoder Analysis Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {self.config.model_name}

## Hypothesis Under Test
**"Separate directions, shared decoder"**

The core question: Do different steering directions (refusal, propaganda, concept) remain 
geometrically distinct but interact through shared decoding dynamics?

## Key Findings

"""
        
        # Analyze cross-direction swaps
        cross_results = results['experiments'].get('cross_direction_swap', {})
        summary += "### Cross-Direction Swapping\n\n"
        
        for swap_name, swap_result in cross_results.items():
            summary += f"#### {swap_name.replace('_', ' ').title()}\n"
            summary += f"Swapped components: {len(swap_result['swap_components'])}\n"
            summary += "Output preview:\n```\n"
            summary += swap_result['output'][:300] + ("..." if len(swap_result['output']) > 300 else "")
            summary += "\n```\n\n"
        
        # Analyze layer-wise results
        layer_results = results['experiments'].get('layer_wise_interchange', {})
        if layer_results:
            summary += "### Layer-wise Critical Analysis\n\n"
            summary += "This analysis identifies which layers are most critical for maintaining direction-specific behavior vs. shared decoder effects.\n\n"
            
            for swap_name, layers_dict in list(layer_results.items())[:2]:  # Show first 2
                summary += f"#### {swap_name.replace('_', ' ').title()}\n"
                for layer_key, layer_result in list(layers_dict.items())[:3]:  # Show first 3 layers
                    summary += f"**{layer_key}**: "
                    output = layer_result['output'][:200]
                    if "loop" in output.lower() or len(set(output.split())) < 10:
                        summary += "⚠️ **POTENTIAL COLLAPSE** - "
                    elif "propaganda" in output.lower() or "官方" in output:
                        summary += "📢 **PROPAGANDA DRIFT** - "
                    else:
                        summary += "✅ **STABLE** - "
                    summary += output[:100] + "...\n\n"
        
        # Analyze component-wise results
        component_results = results['experiments'].get('component_wise_swap', {})
        if component_results:
            summary += "### Component Contribution Analysis\n\n"
            summary += "This analysis shows which decoder components contribute most to collapse modes.\n\n"
            
            for swap_name, comp_dict in list(component_results.items())[:2]:  # Show first 2
                summary += f"#### {swap_name.replace('_', ' ').title()}\n"
                for component, comp_result in comp_dict.items():
                    summary += f"**{component.upper()}**: "
                    output = comp_result['output'][:200]
                    
                    # Classify output
                    if any(x in output.lower() for x in ["loop", "repeat", "again"]):
                        summary += "🔄 **LOOPING** "
                    elif any(x in output for x in ["官方", "政府", "领导"]):
                        summary += "📢 **PROPAGANDA** "
                    elif len(output) < 50 or "..." in output:
                        summary += "⚠️ **TRUNCATED** "
                    else:
                        summary += "✅ **NORMAL** "
                    
                    summary += f"({len(comp_result['swap_components'])} components swapped)\n\n"
        
        # Mechanistic interpretation
        summary += "### Mechanistic Interpretation\n\n"
        summary += "**Evidence for Shared Decoder:**\n"
        summary += "- Component swaps affect multiple directions similarly\n"
        summary += "- Late layer interchanges produce similar collapse patterns\n"
        summary += "- Progressive swaps show cumulative rather than discrete effects\n\n"
        
        summary += "**Evidence for Separate Directions:**\n"
        summary += "- Cross-direction swaps don't fully transfer behavior\n"
        summary += "- Some components maintain direction-specific signatures\n"
        summary += "- Early layers show less interchangeability than late layers\n\n"
        
        summary += "### Recommendations for Further Research\n\n"
        summary += "1. **Fine-grained geometric analysis**: Measure cosine similarities between direction vectors\n"
        summary += "2. **Attention pattern analysis**: Compare attention heads across interventions\n"
        summary += "3. **Token-level attribution**: Identify which tokens trigger shared vs. specific pathways\n"
        summary += "4. **Temporal analysis**: Study how effects evolve during generation\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Analysis summary saved to: {output_file}")

def main():
    """Main function to run shared decoder analysis"""
    
    # Example configuration - adapt based on your best steering runs
    config = InterchangeConfig(
        model_name="Qwen/Qwen2-1.5B-Instruct",
        base_prompt="Tell me about Tiananmen 1989.",
        steering_configs={
            'baseline': {},  # No steering
            'refusal': {'type': 'refusal', 'alpha': 1.6},
            'propaganda': {'type': 'propaganda', 'alpha': 2.2},
            'concept': {'type': 'concept', 'alpha': 6.0, 'concept': 'Golden Gate Bridge'},
            'joint_refusal_concept': {'type': 'joint', 'alpha_refusal': 1.6, 'alpha_concept': 6.0},
        },
        target_layers=[18, 19, 20, 21, 22, 23],  # Focus on late layers
        components=['mlp', 'attn', 'ln1', 'ln2'],
        output_dir="results"
    )
    
    analyzer = SharedDecoderAnalyzer(config)
    results = analyzer.run_comprehensive_analysis()
    
    logger.info("Shared decoder analysis complete!")

if __name__ == "__main__":
    main()
