import train
import inference
import edgenetx
import loss
import MetricsVisualizer
import custom_BiFPN




print("Starting enhanced training with dynamic components...")
history = train()
print("\nStarting enhanced inference...")
final_metrics = inference(history)
            
print("\n======== TRAINING COMPLETE ========")
print("Features implemented:")
print("Dynamic NMS with learnable parameters")
print("Dynamic Head with adaptive weights")
print("Comprehensive metrics visualization")
print("Enhanced post-processing")
print("Detailed performance analysis")
