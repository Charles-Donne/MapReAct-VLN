#!/bin/bash
# VLM Navigation Controller
# VLM自动导航系统：LLM规划 + VLM执行 + 语义建图
# 用法: bash run_r2r/vlm_navigation.sh [episode_id] [max_steps]

set -e
trap 'echo "❌ 错误：脚本在第 $LINENO 行失败"; exit 1' ERR

# 环境变量
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error

# 参数解析
EPISODE_ID=${1:-0}
MAX_STEPS=${2:-500}

# 参数验证
if ! [[ "$EPISODE_ID" =~ ^[0-9]+$ ]]; then
    echo "❌ episode_id必须是正整数: $EPISODE_ID"
    exit 1
fi

if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || [ "$MAX_STEPS" -lt 1 ]; then
    echo "❌ max_steps必须大于0: $MAX_STEPS"
    exit 1
fi

# 配置路径
CONFIG_FILE="vlnce_baselines/config/exp1.yaml"
RESULTS_DIR="data/vlm_navigation"
LLM_CONFIG="vlnce_baselines/vlm/llm_config.yaml"
VLM_CONFIG="vlnce_baselines/vlm/vlm_config.yaml"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Habitat配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$LLM_CONFIG" ]; then
    echo "⚠️  LLM配置文件不存在: $LLM_CONFIG"
    echo "   请从 llm_config.yaml.template 复制并配置"
fi

if [ ! -f "$VLM_CONFIG" ]; then
    echo "⚠️  VLM配置文件不存在: $VLM_CONFIG"
    echo "   请从 vlm_config.yaml.template 复制并配置"
fi

# 环境检查
if ! command -v python &> /dev/null; then
    echo "❌ 未找到Python环境"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "⚠️  未检测到GPU，将使用CPU模式"
fi

# 打印配置信息
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           VLM Navigation Controller                        ║"
echo "║       LLM Planning + VLM Action Execution                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 配置: Episode $EPISODE_ID | 最大步数 $MAX_STEPS"
echo "📁 结果: $RESULTS_DIR/episode_$EPISODE_ID/"
echo ""
echo "🤖 模型配置:"
echo "   LLM: $LLM_CONFIG"
echo "   VLM: $VLM_CONFIG"
echo ""
echo "🔄 工作流程:"
echo "   1. 360°环视建图 + 收集4方向图像"
echo "   2. LLM生成子任务规划"
echo "   3. VLM循环执行动作"
echo "   4. 验证子任务完成并重规划"
echo ""
echo "💾 输出目录:"
echo "   rgb/          - RGB观测图像"
echo "   global_map/   - 全局语义地图"
echo "   detection/    - 检测结果"
echo "   vlm/          - VLM相关文件"
echo "     observations/ - 4方向观察图像"
echo "     subtasks/     - 子任务JSON"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "🚀 启动中..."
START_TIME=$(date +%s)

set +e
CUDA_VISIBLE_DEVICES=0 python vlm_navigation.py \
    --exp-config "$CONFIG_FILE" \
    --episode-id "$EPISODE_ID" \
    --max-steps "$MAX_STEPS" \
    --results-dir "$RESULTS_DIR" \
    --llm-config "$LLM_CONFIG" \
    --vlm-config "$VLM_CONFIG" \
    --max-subtask-steps 50 \
    --verify-interval 10

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
    VLM_COUNT=$(find "$OUTPUT_DIR/vlm" -name "*.json" 2>/dev/null | wc -l)
    echo "📁 $OUTPUT_DIR/"
    echo "   RGB: $RGB_COUNT | Map: $MAP_COUNT | VLM: $VLM_COUNT"
else
    echo "⚠️  输出目录未创建"
fi

echo "════════════════════════════════════════════════════════════"

exit $EXIT_CODE
