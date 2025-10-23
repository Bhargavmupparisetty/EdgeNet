# =========================================================
#   COMPREHENSIVE VISUALIZATION CLASS
# =========================================================
class MetricsVisualizer:
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'map': [],
            'map_50': [],
            'map_75': [],
            'mar_100': [],
            'learning_rate': [],
            'nms_threshold': [],
            'class_weights': []
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def plot_training_curves(self, figsize=(20, 12)):
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        
        # Training Loss
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss', color='blue')
        if self.metrics_history['val_loss']:
            axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # mAP Metrics
        axes[0, 1].plot(self.metrics_history['map'], label='mAP', marker='o', color='green')
        axes[0, 1].plot(self.metrics_history['map_50'], label='mAP@0.5', marker='s', color='blue')
        axes[0, 1].plot(self.metrics_history['map_75'], label='mAP@0.75', marker='^', color='red')
        axes[0, 1].set_title('mAP Metrics')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average Recall
        axes[0, 2].plot(self.metrics_history['mar_100'], label='AR@100', marker='d', color='purple')
        axes[0, 2].set_title('Average Recall')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('AR')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        if self.metrics_history['learning_rate']:
            axes[0, 3].plot(self.metrics_history['learning_rate'], color='orange')
            axes[0, 3].set_title('Learning Rate Schedule')
            axes[0, 3].set_xlabel('Epoch')
            axes[0, 3].set_ylabel('LR')
            axes[0, 3].grid(True, alpha=0.3)
        
        # NMS Threshold Evolution
        if self.metrics_history['nms_threshold']:
            axes[1, 0].plot(self.metrics_history['nms_threshold'], color='brown')
            axes[1, 0].set_title('Dynamic NMS Threshold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('NMS Threshold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Class Weights Evolution
        if self.metrics_history['class_weights']:
            class_weights = np.array(self.metrics_history['class_weights'])
            for i in range(class_weights.shape[1]):
                axes[1, 1].plot(class_weights[:, i], label=f'Class {i}')
            axes[1, 1].set_title('Dynamic Class Weights')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Performance Improvement
        if len(self.metrics_history['map_50']) > 1:
            improvement = np.diff(self.metrics_history['map_50'])
            axes[1, 2].plot(improvement, color='teal')
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].set_title('mAP@0.5 Improvement per Epoch')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Improvement')
            axes[1, 2].grid(True, alpha=0.3)
        
        # Final metrics distribution
        if self.metrics_history['map']:
            final_metrics = {
                'mAP': self.metrics_history['map'][-1],
                'mAP@0.5': self.metrics_history['map_50'][-1],
                'mAP@0.75': self.metrics_history['map_75'][-1],
                'AR@100': self.metrics_history['mar_100'][-1]
            }
            axes[1, 3].bar(final_metrics.keys(), final_metrics.values(), 
                          color=['green', 'blue', 'red', 'purple'])
            axes[1, 3].set_title('Final Metrics')
            axes[1, 3].set_ylabel('Score')
            axes[1, 3].set_ylim(0, 1)
            for i, (k, v) in enumerate(final_metrics.items()):
                axes[1, 3].text(i, v + 0.02, f'{v:.3f}', ha='center')
            axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self):
        # Create correlation matrix of metrics
        metrics_df = pd.DataFrame({
            'mAP': self.metrics_history['map'],
            'mAP@0.5': self.metrics_history['map_50'],
            'mAP@0.75': self.metrics_history['map_75'],
            'AR@100': self.metrics_history['mar_100']
        })
        
        plt.figure(figsize=(8, 6))
        correlation_matrix = metrics_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Metrics Correlation Matrix')
        plt.tight_layout()
        plt.show()
