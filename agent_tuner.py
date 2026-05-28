"""
LLM Agent for PID auto-tuning (a/b/c/d)
Compatible with DeepSeek API, openai>=1.0.0
Author: Group Member 3
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil

# ============================================
# OpenAI / DeepSeek
# ============================================
USE_LLM = True

try:
    from openai import OpenAI
except:
    USE_LLM = False

# ============================================
# DeepSeek API
# ============================================
DEEPSEEK_API_KEY = "你的apikey"

MODEL_NAME = "deepseek-chat"

client = None

if USE_LLM:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

# ============================================
# 清空旧图片
# ============================================
SIM_RESULT_DIR = "simulation_results"

if os.path.exists(SIM_RESULT_DIR):
    shutil.rmtree(SIM_RESULT_DIR)

os.makedirs(SIM_RESULT_DIR, exist_ok=True)

# ============================================
# 导入评估函数
# ============================================
from env_simulator import (
    evaluate_problem_1_pd,
    evaluate_problem_2_pi,
    evaluate_problem_3_pid,
    evaluate_problem_4_pid_freq
)

# ============================================
# 达标判断
# ============================================
def is_satisfied(problem, metrics, tol=1e-6):

    if problem == 'a':

        return (
            metrics.get('e_ss_ramp', 1e9) <= 0.000443 + tol and
            metrics.get('overshoot', 1e9) <= 5.0 + tol and
            metrics.get('tr', 1e9) <= 0.005 + tol and
            metrics.get('ts', 1e9) <= 0.005 + tol
        )

    elif problem == 'b':

        return (
            metrics.get('e_ss_acc', 1e9) <= 0.2 + tol and
            metrics.get('overshoot', 1e9) <= 5.0 + tol and
            metrics.get('tr', 1e9) <= 0.01 + tol and
            metrics.get('ts', 1e9) <= 0.02 + tol
        )

    elif problem == 'c':

        return (
            metrics.get('e_ss_acc', 1e9) <= 0.2 + tol and
            metrics.get('overshoot', 1e9) <= 5.0 + tol and
            metrics.get('tr', 1e9) <= 0.005 + tol and
            metrics.get('ts', 1e9) <= 0.005 + tol
        )

    elif problem == 'd':

        return (
            metrics.get('e_ss_acc', 1e9) <= 0.2 + tol and
            metrics.get('phase_margin', -1e9) >= 70.0 - tol and
            metrics.get('Mr', 1e9) <= 1.1 + tol and
            metrics.get('BW', -1e9) >= 1000.0 - tol
        )

    return False


# ============================================
# 评估
# ============================================
def evaluate(problem, params):

    if problem == 'a':
        metrics = evaluate_problem_1_pd(params[0], params[1])

    elif problem == 'b':
        metrics = evaluate_problem_2_pi(params[0], params[1])

    elif problem == 'c':
        metrics = evaluate_problem_3_pid(params[0], params[1], params[2])

    else:
        metrics = evaluate_problem_4_pid_freq(
            params[0],
            params[1],
            params[2]
        )

    files = [
        os.path.join(SIM_RESULT_DIR, f)
        for f in os.listdir(SIM_RESULT_DIR)
        if f.endswith('.png')
    ]

    latest_img = max(files, key=os.path.getmtime) if files else None

    return metrics, latest_img


# ============================================
# 获取规格
# ============================================
def get_specs_text(problem):

    specs = {

        'a':
        "单位斜坡稳态误差≤0.000443, 超调≤5%, 上升时间≤0.005s, 调节时间≤0.005s",

        'b':
        "加速度稳态误差≤0.2, 超调≤5%, 上升时间≤0.01s, 调节时间≤0.02s",

        'c':
        "加速度稳态误差≤0.2, 超调≤5%, 上升时间≤0.005s, 调节时间≤0.005s",

        'd':
        "加速度稳态误差≤0.2, 相位裕度≥70°, 谐振峰值≤1.1, 带宽≥1000rad/s"
    }

    return specs[problem]


# ============================================
# 启发式调参
# ============================================
def heuristic_tune(problem, last_params, last_metrics):

    # ====================================
    # Problem a
    # ====================================
    if problem == 'a':

        Kp, Kd = last_params

        tr = float(last_metrics.get('tr', 1))
        ts = float(last_metrics.get('ts', 1))
        ov = float(last_metrics.get('overshoot', 0))
        ess = float(last_metrics.get('e_ss_ramp', 1))

        # 加速系统
        if tr > 0.005:
            Kp *= 2.5

        # 缩短调节时间
        if ts > 0.005:
            Kp *= 2.0
            Kd *= 1.3

        # 抑制超调
        if ov > 5:
            Kd *= 1.8
            Kp *= 0.9

        # 降低稳态误差
        if ess > 0.000443:
            Kp *= 1.5

        Kp = np.clip(Kp, 0.1, 5000)
        Kd = np.clip(Kd, 0.01, 1000)

        return [float(Kp), float(Kd)]

    # ====================================
    # Problem b
    # ====================================
    elif problem == 'b':

        Kp, Ki = last_params

        tr = last_metrics.get('tr', 1)
        ov = last_metrics.get('overshoot', 0)
        ess = last_metrics.get('e_ss_acc', 1)

        if tr > 0.01:
            Kp *= 1.4

        if ov > 5:
            Kp *= 0.92

        if ess > 0.2:
            Ki *= 1.3

        return [
            float(np.clip(Kp, 0.1, 1000)),
            float(np.clip(Ki, 0.01, 500))
        ]

    # ====================================
    # Problem c
    # ====================================
    elif problem == 'c':

        Kp, Ki, Kd = last_params

        tr = float(last_metrics.get('tr', 1))
        ts = float(last_metrics.get('ts', 1))
        ov = float(last_metrics.get('overshoot', 0))
        ess = float(last_metrics.get('e_ss_acc', 1))

        if tr > 0.005:
            Kp *= 1.8
            Kd *= 1.3

        if ts > 0.005:
            Kp *= 1.5
            Kd *= 1.5

        if ov > 5:
            Kd *= 1.8
            Kp *= 0.92

        if ess > 0.2:
            Ki *= 1.25

        Kp = np.clip(Kp, 1, 5000)
        Ki = np.clip(Ki, 0.1, 1000)
        Kd = np.clip(Kd, 0.01, 1000)

        return [float(Kp), float(Ki), float(Kd)]

    # ====================================
    # Problem d
    # ====================================
    else:

        Kp, Ki, Kd = last_params

        bw = float(last_metrics.get('BW', 0))
        pm = float(last_metrics.get('phase_margin', 0))
        mr = float(last_metrics.get('Mr', 10))
        ess = float(last_metrics.get('e_ss_acc', 1))

        # BW 不够
        if bw < 1000:

            ratio = (1000 - bw) / 1000

            Kp *= (1.2 + ratio)
            Kd *= (1.15 + 0.5 * ratio)

        # 相位裕度不够
        if pm < 70:

            ratio = (70 - pm) / 70

            Kp *= (0.92 - 0.1 * ratio)
            Kd *= (1.15 + ratio)

        # Mr 太大
        if mr > 1.1:

            ratio = mr - 1.1

            Kp *= (0.90 - 0.05 * ratio)
            Kd *= (1.20 + 0.2 * ratio)

        # 稳态误差太大
        if ess > 0.2:

            ratio = ess / 0.2

            Ki *= (1.15 + 0.1 * ratio)

        Kp = np.clip(Kp, 1, 800)
        Ki = np.clip(Ki, 0.1, 300)
        Kd = np.clip(Kd, 0.01, 80)

        return [float(Kp), float(Ki), float(Kd)]


# ============================================
# LLM 调参
# ============================================
def ask_llm_for_params(problem, history):

    if not USE_LLM:
        return heuristic_tune(
            problem,
            history[-1]['params'],
            history[-1]['metrics']
        )

    system_prompt = f"""
你是 PID 控制专家。

问题类型: {problem}

规格要求:
{get_specs_text(problem)}

请根据历史记录输出下一组参数。

只允许输出 JSON。

问题a:
{{"Kp": xxx, "Kd": xxx}}

问题b:
{{"Kp": xxx, "Ki": xxx}}

问题c/d:
{{"Kp": xxx, "Ki": xxx, "Kd": xxx}}
"""

    user_msg = ""

    for h in history:

        user_msg += f"""
迭代 {h['iteration']}
参数: {h['params']}
指标: {h['metrics']}
"""

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.4,
            max_tokens=200
        )

        reply = response.choices[0].message.content.strip()

        # 安全 JSON 解析
        start = reply.find("{")
        end = reply.rfind("}")

        if start != -1 and end != -1:
            reply = reply[start:end+1]

        data = json.loads(reply)

        if problem == 'a':

            return [
                float(data.get("Kp", 1.0)),
                float(data.get("Kd", 0.1))
            ]

        elif problem == 'b':

            return [
                float(data.get("Kp", 1.0)),
                float(data.get("Ki", 0.5))
            ]

        else:

            return [
                float(data.get("Kp", 10.0)),
                float(data.get("Ki", 5.0)),
                float(data.get("Kd", 1.0))
            ]

    except Exception as e:

        print("LLM 调用失败:", e)

        return heuristic_tune(
            problem,
            history[-1]['params'],
            history[-1]['metrics']
        )


# ============================================
# 主调参函数
# ============================================
def tune_pid(problem, max_iter=12, verbose=True):

    # 初始参数
    if problem == 'a':
        params = [1.0, 0.05]

    elif problem == 'b':
        params = [1.0, 0.5]

    elif problem == 'c':
        params = [50.0, 10.0, 1.0]

    else:
        params = [180.0, 35.0, 12.0]

    history = []

    best_params = params.copy()
    best_metrics = None
    best_satisfied = False

    for i in range(max_iter):

        metrics, img_path = evaluate(problem, params)

        satisfied = is_satisfied(problem, metrics)

        history.append({

            'iteration': i + 1,
            'params': params.copy(),
            'metrics': metrics,
            'satisfied': satisfied,
            'img_path': img_path
        })

        if verbose:

            print("\n============================")
            print(f"迭代 {i+1}")
            print("============================")

            print("参数:", params)

            print("指标:")

            for k, v in metrics.items():
                print(f"{k}: {v}")

            print("达标:", satisfied)

        # 达标
        if satisfied:

            best_params = params.copy()
            best_metrics = metrics
            best_satisfied = True

            break

        # 保存最好结果
        if best_metrics is None:

            best_metrics = metrics
            best_params = params.copy()

        else:

            if problem != 'd':

                if metrics.get('ts', 1e9) < best_metrics.get('ts', 1e9):

                    best_metrics = metrics
                    best_params = params.copy()

            else:

                score_new = (
                    metrics.get('BW', 0)
                    - 50 * max(0, metrics.get('Mr', 10) - 1.1)
                    + metrics.get('phase_margin', 0)
                )

                score_old = (
                    best_metrics.get('BW', 0)
                    - 50 * max(0, best_metrics.get('Mr', 10) - 1.1)
                    + best_metrics.get('phase_margin', 0)
                )

                if score_new > score_old:

                    best_metrics = metrics
                    best_params = params.copy()

        # LLM 调参
        params = ask_llm_for_params(problem, history)

        params = [max(0.0, float(p)) for p in params]

    # ============================================
    # 保存结果图
    # ============================================
    fig_dir = "agent_figures"

    os.makedirs(fig_dir, exist_ok=True)

    _, final_img = evaluate(problem, best_params)

    final_path = None

    if final_img and os.path.exists(final_img):

        final_path = os.path.join(
            fig_dir,
            f'final_response_{problem}.png'
        )

        shutil.copy(final_img, final_path)

    return {

        'best_params': best_params,

        'best_metrics': best_metrics,

        'success': best_satisfied,

        'figures': {
            'final_response': final_path
        }
    }


# ============================================
# 命令行入口
# ============================================
if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("用法:")
        print("python agent_tuner.py [a|b|c|d]")

        sys.exit(1)

    prob = sys.argv[1].lower()

    if prob not in ['a', 'b', 'c', 'd']:

        print("问题必须是 a/b/c/d")

        sys.exit(1)

    print(f"\n========== 开始调参：问题 {prob} ==========\n")

    result = tune_pid(
        problem=prob,
        max_iter=12,
        verbose=True
    )

    print("\n===================================")
    print("最终结果")
    print("===================================")

    print("达标:", result['success'])

    print("最优参数:", result['best_params'])

    print("最优指标:")

    for k, v in result['best_metrics'].items():
        print(f"{k}: {v}")

    print("\n图片保存位置:")

    print(result['figures'])
