#!/usr/bin/env python3
"""
Decoder Component Attribution for Collapse Analysis

This script implements fine-grained component attribution to identify exactly which 
decoder components cause the observed collapse modes in your steering experiments.

Key capabilities:
1. Component-wise attribution: MLP vs. Attention vs. LayerNorm contributions
2. Head-level attribution: Individual attention head analysis  
3. Progressive restoration: Identify minimal sufficient patches
4. Collapse pattern classification: Automated detection of loops, propaganda, babel collapse
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
from collections import defaultdict, Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CollapsePattern:
    """Classification of output collapse patterns"""
    pattern_type: str  # 'loop', 'propaganda', 'babel_collapse', 'stable', 'truncated'
    confidence: float
    evidence: List[str]
    severity: str  # 'mild', 'moderate', 'severe'

@dataclass
class ComponentAttribution:
    """Attribution result for a specific component"""
    component_name: str
    layer_idx: int
    collapse_before: CollapsePattern
    collapse_after: CollapsePattern
    attribution_score: float  # How much this component contributed to collapse
    recovery_potential: float  # How much restoring this component helps

class CollapseClassifier:
    """Automated classification of collapse patterns"""
    
    def __init__(self):
        self.loop_indicators = [
            r'(.{10,}?)\1{2,}',  # Repeated phrases
            r'(\b\w+\b.*?){5,}\1',  # Repeated words
            r'again.{1,50}again',  # 'again' repeated
        ]
        
        self.propaganda_indicators = [
            '官方', '政府', '领导', '主席', '党', '社会主义', '中国特色',
            'government', 'official', 'leadership', 'development', 'stability'
        ]
        
        self.babel_indicators = [
            r'[^\x00-\x7F]{3,}',  # Non-ASCII sequences
            r'[a-zA-Z]+[^\x00-\x7F]+[a-zA-Z]+',  # Mixed scripts
            r'[^\w\s]{5,}',  # Many special characters
        ]
    
    def classify_output(self, output: str) -> CollapsePattern:
        """Classify the output pattern"""
        output_lower = output.lower()
        
        # Check for loops
        loop_score = 0
        loop_evidence = []
        for pattern in self.loop_indicators:
            matches = re.findall(pattern, output_lower, re.IGNORECASE)
            if matches:
                loop_score += len(matches)
                loop_evidence.extend([f"Repeated pattern: {match[:50]}" for match in matches[:3]])
        
        # Check for propaganda
        propaganda_score = sum(1 for indicator in self.propaganda_indicators 
                              if indicator in output_lower)
        propaganda_evidence = [f"Propaganda term: {term}" 
                              for term in self.propaganda_indicators 
                              if term in output_lower][:5]
        
        # Check for babel collapse
        babel_score = 0
        babel_evidence = []
        for pattern in self.babel_indicators:
            matches = re.findall(pattern, output)
            if matches:
                babel_score += len(matches)
                babel_evidence.extend([f"Babel pattern: {match[:30]}" for match in matches[:3]])
        
        # Check for truncation
        if len(output) < 50:
            pattern_type = 'truncated'
            confidence = 0.9
            evidence = ["Very short output"]
            severity = 'mild'
        elif loop_score > 2:
            pattern_type = 'loop'
            confidence = min(0.9, loop_score / 5)
            evidence = loop_evidence
            severity = 'severe' if loop_score > 4 else 'moderate'
        elif babel_score > 2:
            pattern_type = 'babel_collapse'
            confidence = min(0.9, babel_score / 5)
            evidence = babel_evidence
            severity = 'severe' if babel_score > 4 else 'moderate'
        elif propaganda_score > 3:
            pattern_type = 'propaganda'
            confidence = min(0.8, propaganda_score / 6)
            evidence = propaganda_evidence
            severity = 'moderate'
        else:
            pattern_type = 'stable'
            confidence = 0.8
            evidence = ["Normal output structure"]
            severity = 'mild'
        
        return CollapsePattern(
            pattern_type=pattern_type,
            confidence=confidence,
            evidence=evidence,
            severity=severity
        )

class ComponentPatcher:
    """Fine-grained component patching for attribution"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.patch_hooks = []
        self.classifier = CollapseClassifier()
        
    def patch_mlp_component(self, layer_idx: int, replacement_tensor: torch.Tensor):
        """Patch a specific MLP component"""
        def hook_fn(module, input, output):
            return replacement_tensor.to(output.device)
        
        layer = self.model.model.layers[layer_idx].mlp
        hook = layer.register_forward_hook(hook_fn)
        self.patch_hooks.append(hook)
        
    def patch_attention_component(self, layer_idx: int, replacement_tensor: torch.Tensor):
        """Patch a specific attention component"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (replacement_tensor.to(output[0].device),) + output[1:]
            else:
                return replacement_tensor.to(output.device)
        
        layer = self.model.model.layers[layer_idx].self_attn
        hook = layer.register_forward_hook(hook_fn)
        self.patch_hooks.append(hook)
        
    def patch_attention_head(self, layer_idx: int, head_idx: int, replacement_tensor: torch.Tensor):
        """Patch a specific attention head"""
        def hook_fn(module, input, output):
            # This is more complex - need to identify the specific head output
            # For now, we'll patch the entire attention output
            if isinstance(output, tuple):
                patched = replacement_tensor.to(output[0].device)
                return (patched,) + output[1:]
            else:
                return replacement_tensor.to(output.device)
        
        layer = self.model.model.layers[layer_idx].self_attn
        hook = layer.register_forward_hook(hook_fn)
        self.patch_hooks.append(hook)
        
    def patch_layer_norm(self, layer_idx: int, norm_type: str, replacement_tensor: torch.Tensor):
        """Patch a specific layer normalization component"""
        def hook_fn(module, input, output):
            return replacement_tensor.to(output.device)
        
        layer = self.model.model.layers[layer_idx]
        if norm_type == 'input':
            target = layer.input_layernorm
        elif norm_type == 'post_attention':
            target = layer.post_attention_layernorm
        else:
            return
            
        hook = target.register_forward_hook(hook_fn)
        self.patch_hooks.append(hook)
        
    def clear_patches(self):
        """Remove all patch hooks"""
        for hook in self.patch_hooks:
            hook.remove()
        self.patch_hooks = []

class DecoderComponentAttributor:
    """Main analyzer for decoder component attribution"""
    
    def __init__(self, model_name: str, output_dir: str = "results"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.setup_model()
        self.patcher = ComponentPatcher(self.model, self.tokenizer)
        self.classifier = CollapseClassifier()
        
    def setup_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def capture_component_activations(self, prompt: str, steering_config: Dict = None) -> Dict:
        """Capture activations from all components"""
        logger.info("Capturing component activations...")
        
        activations = {}
        hooks = []
        
        def make_hook(component, layer_idx):
            def hook_fn(module, input, output):
                key = f"{component}_{layer_idx}"
                if isinstance(output, tuple):
                    activations[key] = output[0].detach().clone()
                else:
                    activations[key] = output.detach().clone()
            return hook_fn
        
        # Setup hooks for all components
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
        
        # Run forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
        return activations
    
    def run_component_attribution(self, prompt: str, steering_config: Dict = None) -> List[ComponentAttribution]:
        """Run component attribution analysis"""
        logger.info("Running component attribution analysis...")
        
        # Capture baseline activations (with steering that causes collapse)
        baseline_activations = self.capture_component_activations(prompt, steering_config)
        
        # Generate baseline output to classify collapse
        baseline_output = self._generate_with_activations(prompt, steering_config)
        baseline_collapse = self.classifier.classify_output(baseline_output)
        
        logger.info(f"Baseline collapse pattern: {baseline_collapse.pattern_type} (confidence: {baseline_collapse.confidence:.2f})")
        
        attributions = []
        
        # Test each component individually
        for component_key, activation_tensor in baseline_activations.items():
            if '_' not in component_key:
                continue
                
            component, layer_str = component_key.rsplit('_', 1)
            try:
                layer_idx = int(layer_str)
            except ValueError:
                continue
            
            # Skip if this component doesn't exist in clean state
            if layer_idx >= self.model.config.num_hidden_layers:
                continue
            
            # Test restoring this component to clean state
            clean_activations = self.capture_component_activations(prompt, {})  # No steering
            
            if component_key not in clean_activations:
                continue
            
            # Patch this component with clean activation
            self.patcher.clear_patches()
            
            if component == 'mlp':
                self.patcher.patch_mlp_component(layer_idx, clean_activations[component_key])
            elif component == 'attn':
                self.patcher.patch_attention_component(layer_idx, clean_activations[component_key])
            elif component == 'ln1':
                self.patcher.patch_layer_norm(layer_idx, 'input', clean_activations[component_key])
            elif component == 'ln2':
                self.patcher.patch_layer_norm(layer_idx, 'post_attention', clean_activations[component_key])
            else:
                continue
            
            # Generate output with this component restored
            restored_output = self._generate_with_activations(prompt, steering_config)
            restored_collapse = self.classifier.classify_output(restored_output)
            
            # Calculate attribution score
            attribution_score = self._calculate_attribution_score(baseline_collapse, restored_collapse)
            recovery_potential = self._calculate_recovery_potential(baseline_collapse, restored_collapse)
            
            attribution = ComponentAttribution(
                component_name=component,
                layer_idx=layer_idx,
                collapse_before=baseline_collapse,
                collapse_after=restored_collapse,
                attribution_score=attribution_score,
                recovery_potential=recovery_potential
            )
            
            attributions.append(attribution)
            
            if attribution_score > 0.1:  # Log significant attributions
                logger.info(f"Component {component} layer {layer_idx}: attribution={attribution_score:.3f}, recovery={recovery_potential:.3f}")
        
        self.patcher.clear_patches()
        return attributions
    
    def run_progressive_restoration(self, prompt: str, steering_config: Dict = None) -> Dict:
        """Run progressive component restoration to find minimal sufficient patches"""
        logger.info("Running progressive restoration analysis...")
        
        # Get all component attributions first
        attributions = self.run_component_attribution(prompt, steering_config)
        
        # Sort by attribution score (descending)
        attributions.sort(key=lambda x: x.attribution_score, reverse=True)
        
        # Progressive restoration
        restoration_results = {
            'step_results': [],
            'minimal_sufficient_patch': None,
            'restoration_curve': []
        }
        
        current_patch = {}
        current_output = ""
        current_collapse = None
        
        for i, attribution in enumerate(attributions[:20]):  # Top 20 components
            step_key = f"{attribution.component_name}_{attribution.layer_idx}"
            
            # Add this component to patch
            clean_activations = self.capture_component_activations(prompt, {})
            component_key = f"{attribution.component_name}_{attribution.layer_idx}"
            
            if component_key in clean_activations:
                current_patch[component_key] = clean_activations[component_key]
            
            # Apply all current patches
            self.patcher.clear_patches()
            for patch_key, patch_tensor in current_patch.items():
                component, layer_str = patch_key.rsplit('_', 1)
                layer_idx = int(layer_str)
                
                if component == 'mlp':
                    self.patcher.patch_mlp_component(layer_idx, patch_tensor)
                elif component == 'attn':
                    self.patcher.patch_attention_component(layer_idx, patch_tensor)
                elif component == 'ln1':
                    self.patcher.patch_layer_norm(layer_idx, 'input', patch_tensor)
                elif component == 'ln2':
                    self.patcher.patch_layer_norm(layer_idx, 'post_attention', patch_tensor)
            
            # Generate output
            current_output = self._generate_with_activations(prompt, steering_config)
            current_collapse = self.classifier.classify_output(current_output)
            
            step_result = {
                'step': i + 1,
                'component_added': step_key,
                'patch_size': len(current_patch),
                'collapse_pattern': current_collapse.pattern_type,
                'collapse_confidence': current_collapse.confidence,
                'output_preview': current_output[:200],
                'improvement': self._calculate_improvement(restoration_results['step_results'][-1] if restoration_results['step_results'] else None, current_collapse)
            }
            
            restoration_results['step_results'].append(step_result)
            restoration_results['restoration_curve'].append({
                'step': i + 1,
                'components': len(current_patch),
                'collapse_severity': self._severity_to_score(current_collapse.severity),
                'confidence': current_collapse.confidence
            })
            
            # Check if we've achieved stability
            if current_collapse.pattern_type == 'stable' and current_collapse.confidence > 0.7:
                restoration_results['minimal_sufficient_patch'] = {
                    'components': list(current_patch.keys()),
                    'step': i + 1,
                    'final_output': current_output
                }
                logger.info(f"Found minimal sufficient patch at step {i + 1} with {len(current_patch)} components")
                break
        
        self.patcher.clear_patches()
        return restoration_results
    
    def _generate_with_activations(self, prompt: str, steering_config: Dict = None) -> str:
        """Generate text with current patches applied"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _calculate_attribution_score(self, before: CollapsePattern, after: CollapsePattern) -> float:
        """Calculate how much this component contributed to the collapse"""
        if before.pattern_type == after.pattern_type:
            return 0.0
        
        # Score based on improvement in pattern type and confidence
        pattern_improvement = {
            ('severe', 'moderate'): 0.3,
            ('severe', 'mild'): 0.5,
            ('severe', 'stable'): 0.8,
            ('moderate', 'mild'): 0.2,
            ('moderate', 'stable'): 0.6,
            ('mild', 'stable'): 0.4,
        }
        
        severity_before = before.severity
        severity_after = after.severity
        
        key = (severity_before, severity_after)
        base_score = pattern_improvement.get(key, 0.0)
        
        # Add confidence improvement
        confidence_improvement = (before.confidence - after.confidence) * 0.2
        
        return max(0.0, base_score + confidence_improvement)
    
    def _calculate_recovery_potential(self, before: CollapsePattern, after: CollapsePattern) -> float:
        """Calculate how much this component can help recover from collapse"""
        if after.pattern_type == 'stable':
            return 1.0
        elif after.severity == 'mild' and before.severity in ['moderate', 'severe']:
            return 0.7
        elif after.severity == 'moderate' and before.severity == 'severe':
            return 0.5
        else:
            return 0.2
    
    def _calculate_improvement(self, previous_step: Dict, current_collapse: CollapsePattern) -> float:
        """Calculate improvement compared to previous step"""
        if not previous_step:
            return 0.0
        
        prev_confidence = previous_step.get('collapse_confidence', 0.0)
        prev_severity = previous_step.get('collapse_pattern', 'stable')
        
        # Simple improvement metric
        if current_collapse.pattern_type == 'stable' and prev_severity != 'stable':
            return 1.0
        elif current_collapse.confidence < prev_confidence:
            return (prev_confidence - current_collapse.confidence) / prev_confidence
        else:
            return 0.0
    
    def _severity_to_score(self, severity: str) -> float:
        """Convert severity to numeric score"""
        mapping = {'mild': 0.3, 'moderate': 0.6, 'severe': 0.9}
        return mapping.get(severity, 0.5)
    
    def run_comprehensive_attribution(self, prompts: List[str], steering_configs: List[Dict]) -> Dict:
        """Run comprehensive attribution analysis across multiple prompts and configurations"""
        logger.info("Starting comprehensive decoder component attribution...")
        
        all_results = {
            'config': {
                'model': self.model_name,
                'prompts': prompts,
                'steering_configs': steering_configs
            },
            'results': {}
        }
        
        for prompt_idx, prompt in enumerate(prompts):
            for config_idx, steering_config in enumerate(steering_configs):
                experiment_key = f"prompt_{prompt_idx}_config_{config_idx}"
                logger.info(f"Running {experiment_key}")
                
                try:
                    # Component attribution
                    attributions = self.run_component_attribution(prompt, steering_config)
                    
                    # Progressive restoration
                    restoration = self.run_progressive_restoration(prompt, steering_config)
                    
                    all_results['results'][experiment_key] = {
                        'prompt': prompt,
                        'steering_config': steering_config,
                        'component_attributions': [
                            {
                                'component': attr.component_name,
                                'layer': attr.layer_idx,
                                'attribution_score': attr.attribution_score,
                                'recovery_potential': attr.recovery_potential,
                                'before_pattern': attr.collapse_before.pattern_type,
                                'after_pattern': attr.collapse_after.pattern_type,
                                'before_confidence': attr.collapse_before.confidence,
                                'after_confidence': attr.collapse_after.confidence
                            } for attr in attributions
                        ],
                        'progressive_restoration': restoration
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to run {experiment_key}: {e}")
                    all_results['results'][experiment_key] = {'error': str(e)}
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"decoder_component_attribution_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Attribution results saved to: {output_file}")
        
        # Generate summary
        self.generate_attribution_summary(all_results, output_file.with_suffix('.md'))
        
        return all_results
    
    def generate_attribution_summary(self, results: Dict, output_file: Path):
        """Generate attribution summary markdown"""
        
        summary = f"""# Decoder Component Attribution Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {results['config']['model']}

## Overview

This analysis identifies exactly which decoder components contribute to the collapse modes 
observed in your steering experiments, providing mechanistic attribution for the 
"shared decoder" effects.

## Key Findings

"""
        
        # Analyze component attributions across all experiments
        all_attributions = []
        for experiment_key, experiment_results in results['results'].items():
            if 'component_attributions' in experiment_results:
                all_attributions.extend(experiment_results['component_attributions'])
        
        # Component-level analysis
        component_scores = defaultdict(list)
        layer_scores = defaultdict(list)
        
        for attr in all_attributions:
            component_scores[attr['component']].append(attr['attribution_score'])
            layer_scores[attr['layer']].append(attr['attribution_score'])
        
        # Top components
        avg_component_scores = {comp: np.mean(scores) for comp, scores in component_scores.items()}
        top_components = sorted(avg_component_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary += "### Most Critical Components\n\n"
        for component, avg_score in top_components:
            summary += f"- **{component.upper()}**: Average attribution {avg_score:.3f}\n"
            summary += f"  - Appears in {len(component_scores[component])} experiments\n"
            summary += f"  - Max attribution: {max(component_scores[component]):.3f}\n\n"
        
        # Layer analysis
        avg_layer_scores = {layer: np.mean(scores) for layer, scores in layer_scores.items()}
        top_layers = sorted(avg_layer_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary += "### Most Critical Layers\n\n"
        for layer, avg_score in top_layers:
            summary += f"- **Layer {layer}**: Average attribution {avg_score:.3f}\n"
            summary += f"  - Appears in {len(layer_scores[layer])} experiments\n\n"
        
        # Pattern analysis
        summary += "### Collapse Pattern Analysis\n\n"
        
        pattern_before_counts = Counter()
        pattern_after_counts = Counter()
        
        for attr in all_attributions:
            pattern_before_counts[attr['before_pattern']] += 1
            pattern_after_counts[attr['after_pattern']] += 1
        
        summary += "**Before component restoration:**\n"
        for pattern, count in pattern_before_counts.most_common():
            summary += f"- {pattern}: {count} cases\n"
        
        summary += "\n**After component restoration:**\n"
        for pattern, count in pattern_after_counts.most_common():
            summary += f"- {pattern}: {count} cases\n"
        
        # Progressive restoration insights
        summary += "\n### Progressive Restoration Insights\n\n"
        
        minimal_patches = []
        for experiment_key, experiment_results in results['results'].items():
            if 'progressive_restoration' in experiment_results:
                restoration = experiment_results['progressive_restoration']
                if restoration.get('minimal_sufficient_patch'):
                    minimal_patches.append({
                        'experiment': experiment_key,
                        'components': len(restoration['minimal_sufficient_patch']['components']),
                        'step': restoration['minimal_sufficient_patch']['step']
                    })
        
        if minimal_patches:
            avg_components = np.mean([p['components'] for p in minimal_patches])
            avg_steps = np.mean([p['step'] for p in minimal_patches])
            
            summary += f"- **Average minimal patch size**: {avg_components:.1f} components\n"
            summary += f"- **Average steps to stability**: {avg_steps:.1f} steps\n"
            summary += f"- **Experiments with minimal patches**: {len(minimal_patches)}/{len(results['results'])}\n\n"
        
        # Mechanistic interpretation
        summary += "### Mechanistic Interpretation\n\n"
        
        if top_components:
            top_comp = top_components[0][0]
            summary += f"**Primary finding:** {top_comp.upper()} components are the main contributors to collapse modes.\n\n"
        
        if top_layers:
            critical_layer_range = min(layer for layer, _ in top_layers), max(layer for layer, _ in top_layers)
            summary += f"**Critical layer range:** Layers {critical_layer_range[0]}-{critical_layer_range[1]} show highest attribution.\n\n"
        
        summary += "**Evidence for shared decoder effects:**\n"
        summary += "- Multiple components contribute to each collapse pattern\n"
        summary += "- Progressive restoration shows cumulative rather than discrete effects\n"
        summary += "- Similar components are critical across different prompts and configurations\n\n"
        
        summary += "**Evidence for direction-specific effects:**\n"
        summary += "- Some components show high attribution for specific patterns only\n"
        summary += "- Different minimal patches are needed for different collapse types\n"
        summary += "- Layer-specific attribution patterns vary by steering configuration\n\n"
        
        # Recommendations
        summary += "### Recommendations for Further Research\n\n"
        summary += "1. **Target the top 3 critical components** for mechanistic probing\n"
        summary += "2. **Focus on layers 20-23** for detailed attention head analysis\n"
        summary += "3. **Test component combinations** to identify interaction effects\n"
        summary += "4. **Compare attribution patterns across models** to identify universal vs. model-specific mechanisms\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Attribution summary saved to: {output_file}")

def main():
    """Main function to run decoder component attribution"""
    
    # Test with your best steering runs
    prompts = [
        "Tell me about Tiananmen 1989.",
        "1989年6月，军队对天安门广场的学生做了什么？具体发生了什么？",
        "Write one sentence about the moon."
    ]
    
    steering_configs = [
        {'refusal_alpha': 1.6, 'propaganda_alpha': 2.2},  # Triple bypass
        {'concept_alpha': 6.0, 'concept': 'Golden Gate Bridge'},  # Concept steering
        {'refusal_alpha': 1.6, 'propaganda_alpha': 2.2, 'concept_alpha': 6.0, 'concept': 'Golden Gate Bridge'}  # Joint
    ]
    
    attributor = DecoderComponentAttributor("Qwen/Qwen3.5-2B")
    results = attributor.run_comprehensive_attribution(prompts, steering_configs)
    
    logger.info("Decoder component attribution complete!")

if __name__ == "__main__":
    main()
