def inference(history=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeAirportNet().to(device)
    
    if os.path.exists('best_airport_dynamic.pt'):
        model.load_state_dict(torch.load('best_airport_dynamic.pt', map_location=device))
        print("Loaded best model weights")
    else:
        print("No saved model found, using randomly initialized weights")
    
    model.eval()
    
    val_ds = RunwayDataset('/kaggle/input/airports/Major_dataset', 'val', 512)
    val_loader = DataLoader(val_ds, 4, num_workers=2, collate_fn=collate)
    
    metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
    
    # Collect predictions and confidence scores for analysis
    all_confidences = []
    all_box_areas = []
    all_predictions = []
    
    # Enhanced visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    sample_count = 0
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            if sample_count >= 12:
                break
                
            imgs = imgs.to(device)
            
            # Get predictions with NMS
            predictions = model.post_process(
                *model(imgs), 
                img_size=512
            )
            
            # Update metrics
            tgt = []
            for target in targets:
                if target.numel() == 0:
                    tgt.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'labels': torch.empty(0, dtype=torch.long, device=device)
                    })
                else:
                    tgt.append({
                        'boxes': target[:, :4].to(device),
                        'labels': target[:, 4].long().to(device)
                    })
            
            metric.update(predictions, tgt)
            
            # Visualize samples
            for b in range(min(imgs.size(0), 12 - sample_count)):
                if sample_count >= 12:
                    break
                
                # Denormalize image
                img = imgs[b].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                axes[sample_count].imshow(img)
                
                # Draw predictions with confidence scores
                pred_boxes = predictions[b]['boxes'].cpu().numpy()
                pred_scores = predictions[b]['scores'].cpu().numpy()
                pred_labels = predictions[b]['labels'].cpu().numpy()
                
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = box
                    # Color based on confidence
                    color = 'red' if score > 0.7 else 'orange' if score > 0.5 else 'yellow'
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                    axes[sample_count].add_patch(rect)
                    axes[sample_count].text(x1, y1-5, f'{score:.2f}', 
                                          color=color, fontsize=8, weight='bold')
                
                # Draw ground truth
                if targets[b].numel() > 0:
                    gt_boxes = targets[b][:, :4].cpu().numpy()
                    for box in gt_boxes:
                        x1, y1, x2, y2 = box
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor='green', facecolor='none')
                        axes[sample_count].add_patch(rect)
                
                axes[sample_count].set_title(f'Sample {sample_count+1}\nPred: {len(pred_boxes)} | GT: {len(targets[b])}')
                axes[sample_count].axis('off')
                
                # Collect statistics
                all_confidences.extend(pred_scores)
                box_areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in pred_boxes]
                all_box_areas.extend(box_areas)
                all_predictions.append(len(pred_boxes))
                
                sample_count += 1
    
    plt.tight_layout()
    plt.show()
    
    # Compute and display final metrics
    m = metric.compute()
    print('\n======== FINAL METRICS ========')
    for k, v in m.items():
        print(f'{k}: {v:.4f}')
    
    # Additional analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Confidence distribution
    axes[0, 0].hist(all_confidences, bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Confidence Score Distribution')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box area distribution
    axes[0, 1].hist(all_box_areas, bins=50, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Predicted Box Area Distribution')
    axes[0, 1].set_xlabel('Box Area (pixels²)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Predictions per image
    axes[0, 2].hist(all_predictions, bins=20, alpha=0.7, color='lightcoral')
    axes[0, 2].set_title('Predictions per Image')
    axes[0, 2].set_xlabel('Number of Predictions')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Metrics comparison
    metrics_keys = ['map_50', 'map_75', 'map', 'mar_100']
    metrics_values = [m[k].item() for k in metrics_keys]
    bars = axes[1, 0].bar(metrics_keys, metrics_values, 
                         color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Final Validation Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, metrics_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Confidence vs Area scatter
    if len(all_confidences) > 0 and len(all_box_areas) > 0:
        scatter = axes[1, 1].scatter(all_box_areas, all_confidences, alpha=0.6, c='purple')
        axes[1, 1].set_title('Confidence vs Box Area')
        axes[1, 1].set_xlabel('Box Area (pixels²)')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Performance summary
    axes[1, 2].text(0.1, 0.8, f'Total Predictions: {len(all_confidences)}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f'Avg Confidence: {np.mean(all_confidences):.3f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f'Avg Box Area: {np.mean(all_box_areas):.1f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f'Avg Preds/Image: {np.mean(all_predictions):.1f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Performance Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return m
