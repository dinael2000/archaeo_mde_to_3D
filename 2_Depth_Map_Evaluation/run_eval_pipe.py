import time

from evaluation_pipeline import *

start = time.time()

if __name__ == '__main__':
    rgb_dir = r"data"
    depth_dir = r"gt_depthmaps"

    pred_depth_dir = r"1_DA1_Depth_Estimation/results_depth_maps/depth_npy"

    pred_vis_dir = r"1_DA1_Depth_Estimation/results_depth_maps/depth_colored"

    save_vis_dir = r"2_Depth_Map_Evaluation/depth_visualizations"
    save_pred_dir = r"2_Depth_Map_Evaluation/results_depth_maps"

    save_diagnostics_dir = None
    save_diagnostics_dir_post = r'2_Depth_Map_Evaluation/diagnostics_post_alignment'

    debug_dir = r'2_Depth_Map_Evaluation/debug_dir'

    df_metrics, pred_depth_dictionary = evaluate_dataset(rgb_dir=rgb_dir,
                                                         gt_dir=depth_dir,
                                                         pred_dir=pred_depth_dir,
                                                         pred_vis_dir=pred_vis_dir, 
                                                         visualize=True, 
                                                         model_name="DA1", 
                                                         save_vis_dir=save_vis_dir, 
                                                         save_pred_dir=save_pred_dir, 
                                                         save_diagnostics_dir=save_diagnostics_dir,
                                                         save_diagnostics_dir_post=save_diagnostics_dir_post,
                                                         debug_dir=debug_dir)

end = time.time()


print(f"Script took: {end-start}")
