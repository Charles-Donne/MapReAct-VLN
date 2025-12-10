"""
Interactive Navigation Runner
实时键盘控制导航系统
"""
import argparse
from vlnce_baselines.config.default import get_config
from vlnce_baselines.interactive_navigation_controller import InteractiveNavigationController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, required=True, help="配置文件")
    parser.add_argument("--episode-id", type=int, default=0, help="Episode ID")
    parser.add_argument("--results-dir", type=str, default=None, help="结果目录")
    parser.add_argument("--max-steps", type=int, default=500, help="最大步数")
    args = parser.parse_args()
    
    config = get_config(args.exp_config, [])
    
    from vlnce_baselines.config_system import ConfigHelper
    config = ConfigHelper.setup_episode_config(config, [args.episode_id], num_environments=1)
    if args.results_dir:
        config = ConfigHelper.setup_results_dir(config, args.results_dir)
    config = ConfigHelper.setup_navigation_config(config)
    ConfigHelper.print_config_summary(config)
    
    controller = InteractiveNavigationController(config)
    
    controller.reset_episode(episode_id=args.episode_id)
    controller.look_around()
    
    print("\n" + "="*60)
    print("🎮 实时键盘控制")
    print("="*60)
    print("控制: w=前进 a=左转 d=右转 t=切换轨迹 c=清空轨迹")
    print(f"最大步数: {args.max_steps}")
    print("="*60)
    
    step = 0
    stop_action = False
    
    while step < args.max_steps:
        action = controller.get_keyboard_action()
        result = controller.step(action, save_vis=True)
        
        if result['done']:
            print("\nEpisode自动完成")
            break
        
        if action == 0:
            stop_action = True
            if input("\n继续? (y/n): ").strip().lower() != 'y':
                print("用户退出")
                break
        
        step += 1
    
    if step >= args.max_steps:
        print(f"\n达到最大步数 {args.max_steps}")
    
    controller.finish_episode(
        success=result.get('done', False),
        stop_action=stop_action
    )
    controller.close()
    
    print(f"\n✅ 导航完成 | 结果: {config.RESULTS_DIR}/episode_{args.episode_id}/")


if __name__ == "__main__":
    main()
