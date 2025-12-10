#!/bin/bash
# 解决 Git 分支冲突 - 保留服务器上的 API 配置
# 用法: bash fix_git_conflict.sh

set -e

echo "🔧 解决 Git 冲突..."
echo ""

# 1. 设置合并策略
echo "📋 设置合并策略为 rebase..."
git config pull.rebase false

# 2. 备份 API 配置文件（如果存在）
echo "💾 备份 API 配置..."
if [ -f "vlnce_baselines/vlm/llm_config.yaml" ]; then
    cp vlnce_baselines/vlm/llm_config.yaml vlnce_baselines/vlm/llm_config.yaml.backup
    echo "  ✓ LLM 配置已备份"
fi

if [ -f "vlnce_baselines/vlm/vlm_config.yaml" ]; then
    cp vlnce_baselines/vlm/vlm_config.yaml vlnce_baselines/vlm/vlm_config.yaml.backup
    echo "  ✓ VLM 配置已备份"
fi

# 3. 从 Git 追踪中移除 API 配置文件（但保留本地文件）
echo ""
echo "🗑️  从 Git 追踪中移除 API 配置..."
git rm --cached vlnce_baselines/vlm/llm_config.yaml 2>/dev/null || echo "  LLM config 未追踪"
git rm --cached vlnce_baselines/vlm/vlm_config.yaml 2>/dev/null || echo "  VLM config 未追踪"

# 4. 提交移除操作（如果有变更）
if ! git diff --cached --quiet 2>/dev/null; then
    echo ""
    echo "📝 提交移除配置文件的变更..."
    git commit -m "chore: Remove API config files from Git tracking

These files should only exist locally and not be committed."
fi

# 5. 拉取远程更新
echo ""
echo "⬇️  拉取远程更新..."
git pull origin main

# 6. 恢复 API 配置（如果之前备份了）
echo ""
echo "♻️  恢复 API 配置..."
if [ -f "vlnce_baselines/vlm/llm_config.yaml.backup" ]; then
    mv vlnce_baselines/vlm/llm_config.yaml.backup vlnce_baselines/vlm/llm_config.yaml
    echo "  ✓ LLM 配置已恢复"
fi

if [ -f "vlnce_baselines/vlm/vlm_config.yaml.backup" ]; then
    mv vlnce_baselines/vlm/vlm_config.yaml.backup vlnce_baselines/vlm/vlm_config.yaml
    echo "  ✓ VLM 配置已恢复"
fi

echo ""
echo "✅ 完成！现在你的配置文件只在本地存在，不会被提交到 Git"
echo ""
echo "📋 验证："
echo "  git status  # 应该看不到 llm_config.yaml 和 vlm_config.yaml"
echo ""
