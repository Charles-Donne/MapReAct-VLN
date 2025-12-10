#!/bin/bash
# Interactive Navigation Controller
# 实时键盘控制导航系统
# 用法: bash run_r2r/interactive_navigation.sh [episode_id] [max_steps]

set -e
trap 'echo "❌ 错误：脚本在第 $LINENO 行失败"; exit 1' ERR

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error

EPISODE_ID=${1:-0}
MAX_STEPS=${2:-500}

if ! [[ "$EPISODE_ID" =~ ^[0-9]+$ ]]; then
    echo "❌ episode_id必须是正整数: $EPISODE_ID"
    exit 1
fi

if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || [ "$MAX_STEPS" -lt 1 ]; then
    echo "❌ max_steps必须大于0: $MAX_STEPS"
    exit 1
fi

CONFIG_FILE="vlnce_baselines/config/exp1.yaml"
RESULTS_DIR="data/interactive_navigation"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "❌ 未找到Python环境"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "⚠️  未检测到GPU，将使用CPU模式"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║       Interactive Navigation Controller                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 配置: Episode $EPISODE_ID | 最大步数 $MAX_STEPS"
echo "📁 结果: $RESULTS_DIR/episode_$EPISODE_ID/"
echo ""
echo "🎮 控制: w=前进 a=左转 d=右转 t=切换轨迹 c=清空轨迹"
echo "💾 输出: rgb/ global_map/ local_map/ detection/"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "🚀 启动中..."
START_TIME=$(date +%s)

set +e
CUDA_VISIBLE_DEVICES=0 python interactive_navigation.py \
    --exp-config "$CONFIG_FILE" \
    --episode-id "$EPISODE_ID" \
    --max-steps "$MAX_STEPS" \
    --results-dir "$RESULTS_DIR"

EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 导航完成 | 用时 ${DURATION}秒"
else
    echo "❌ 异常退出 (code: $EXIT_CODE)"
fi

OUTPUT_DIR="$RESULTS_DIR/episode_$EPISODE_ID"
if [ -d "$OUTPUT_DIR" ]; then
    RGB_COUNT=$(find "$OUTPUT_DIR/rgb" -name "*.png" 2>/dev/null | wc -l)
    MAP_COUNT=$(find "$OUTPUT_DIR/global_map" -name "*.png" 2>/dev/null | wc -l)
    echo "📁 $OUTPUT_DIR/ | RGB: $RGB_COUNT | Map: $MAP_COUNT"
else
    echo "⚠️  输出目录未创建"
fi

echo "════════════════════════════════════════════════════════════"

exit $EXIT_CODE
