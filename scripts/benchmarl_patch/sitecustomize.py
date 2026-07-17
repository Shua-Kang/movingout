"""Auto-imported at interpreter startup (when this dir is on PYTHONPATH).

torchrl's CSVLogger hardcodes video_format='mp4' and calls
torchvision.io.write_video, which torchvision 0.27.1 removed -> crash during
BenchMARL evaluation. We override CSVExperiment.add_video to save the eval
video as a .pt tensor (torch.save, no torchvision/ffmpeg). Convert to GIF later.
"""
try:
    import os
    import torch
    from torchrl.record.loggers.csv import CSVExperiment

    def _add_video_pt(self, tag, vid_tensor, global_step=None, **kwargs):
        if global_step is None:
            global_step = self.videos_counter[tag]
            self.videos_counter[tag] += 1
        filepath = os.path.join(
            self.log_dir, "videos", "_".join([tag, str(global_step)]) + ".pt"
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(vid_tensor.cpu() if hasattr(vid_tensor, "cpu") else vid_tensor,
                   filepath)

    CSVExperiment.add_video = _add_video_pt
    print("[sitecustomize] patched CSVExperiment.add_video -> .pt format")
except Exception as _e:  # never break interpreter startup
    print(f"[sitecustomize] video patch skipped: {_e}")
