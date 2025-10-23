def train():
    cfg = dict(
        data_dir='/kaggle/input/airports/Major_dataset',
        img_size=512, 
        batch=8, 
        lr=1e-3, 
        epochs=16,
        num_workers=8,
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        save='best_airport_dynamic.pt'
    )
    
    device = torch.device(cfg['device'])
    model = EdgeAirportNet().to(device)
    visualizer = MetricsVisualizer()
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    train_ds = RunwayDataset(cfg['data_dir'], 'train', cfg['img_size'])
    val_ds = RunwayDataset(cfg['data_dir'], 'val', cfg['img_size'])
    
    train_loader = DataLoader(
        train_ds, cfg['batch'], shuffle=True,
        num_workers=cfg['num_workers'], collate_fn=collate, 
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, cfg['batch'], shuffle=False,
        num_workers=cfg['num_workers'], collate_fn=collate, 
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), cfg['lr'], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, cfg['lr'], epochs=cfg['epochs'],
        steps_per_epoch=len(train_loader)
    )
    scaler = torch.cuda.amp.GradScaler()

    best_map50 = 0

    for epoch in range(cfg['epochs']):
        # ---------- TRAINING ----------
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]}')
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                cls_logits, bbox_preds = model(imgs)
                loss = compute_loss(cls_logits, bbox_preds, targets, cfg['img_size'])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_loss / num_batches
        
        # ---------- VALIDATION ----------
        model.eval()
        metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc='Validation'):
                imgs = imgs.to(device, non_blocking=True)
                
                # Get validation loss
                cls_logits, bbox_preds = model(imgs)
                loss = compute_loss(cls_logits, bbox_preds, targets, cfg['img_size'])
                val_loss += loss.item()
                val_batches += 1
                
                # Get predictions with NMS
                model_eval = model.module if hasattr(model, 'module') else model
                predictions = model_eval.post_process(cls_logits, bbox_preds, cfg['img_size'])
                
                # Convert targets for metrics
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

        # Compute metrics
        metric_results = metric.compute()
        avg_val_loss = val_loss / val_batches
        
        # Get dynamic parameters
        model_eval = model.module if hasattr(model, 'module') else model
        nms_threshold = torch.sigmoid(model_eval.nms.nms_thresh).item()
        class_weights = model_eval.nms.class_weights.detach().cpu().numpy()
        
        # Update visualizer
        visualizer.update(
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            map=metric_results['map'].item(),
            map_50=metric_results['map_50'].item(),
            map_75=metric_results['map_75'].item(),
            mar_100=metric_results['mar_100'].item(),
            learning_rate=scheduler.get_last_lr()[0],
            nms_threshold=nms_threshold,
            class_weights=class_weights
        )

        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | '
              f'mAP={metric_results["map"].item():.4f} | mAP50={metric_results["map_50"].item():.4f} | '
              f'mAP75={metric_results["map_75"].item():.4f} | AR={metric_results["mar_100"].item():.4f} | '
              f'NMS_thresh={nms_threshold:.3f}')

        # Save best model
        if metric_results['map_50'].item() > best_map50:
            best_map50 = metric_results['map_50'].item()
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, cfg['save'])
            print(f'New best model saved with mAP50: {best_map50:.4f}')

        metric.reset()

    # Plot comprehensive results
    visualizer.plot_training_curves()
    visualizer.plot_correlation_matrix()
    
    return visualizer.metrics_history
