"""
LLM Agent for PID auto-tuning (a/b/c/d)
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
DEEPSEEK_API_KEY = "your APIKEY"

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
# 安全 step_info
# ============================================
def safe_metrics(metrics):

    new_metrics = {}

    for k, v in metrics.items():

        try:

            if np.isnan(v):
                new_metrics[k] = 1e9
            elif np.isinf(v):
                new_metrics[k] = 1e9
            else:
                new_metrics[k] = float(v)

        except:
            new_metrics[k] = v

    return new_metrics


# ============================================
# 评估
# ============================================
def evaluate(problem, params):

    try:

        # =========================
        # Problem a
        # env_simulator 里只接受 Kd
        # =========================
        if problem == 'a':

            Kd = float(params[0])

            metrics = evaluate_problem_1_pd(Kd)

        # =========================
        # Problem b
        # =========================
        elif problem == 'b':

            metrics = evaluate_problem_2_pi(
                float(params[0]),
                float(params[1])
            )

        # =========================
        # Problem c
        # =========================
        elif problem == 'c':

            metrics = evaluate_problem_3_pid(
                float(params[0]),
                float(params[1]),
                float(params[2])
            )

        # =========================
        # Problem d
        # =========================
        else:

            metrics = evaluate_problem_4_pid_freq(
                float(params[0]),
                float(params[1]),
                float(params[2])
            )

        metrics = safe_metrics(metrics)

    except Exception as e:

        print("系统评估失败:", e)

        # 返回一个极差指标
        if problem == 'a':

            metrics = {
                "overshoot": 1e9,
                "tr": 1e9,
                "ts": 1e9,
                "e_ss_ramp": 1e9
            }

        elif problem in ['b', 'c']:

            metrics = {
                "overshoot": 1e9,
                "tr": 1e9,
                "ts": 1e9,
                "e_ss_acc": 1e9
            }

        else:

            metrics = {
                "e_ss_acc": 1e9,
                "phase_margin": 0,
                "Mr": 1e9,
                "BW": 0
            }

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
    # 这里只调 Kd
    # ====================================
    if problem == 'a':

        Kd = float(last_params[0])

        tr = float(last_metrics.get('tr', 1))
        ts = float(last_metrics.get('ts', 1))
        ov = float(last_metrics.get('overshoot', 0))

        if tr > 0.005:
            Kd *= 1.8

        if ts > 0.005:
            Kd *= 1.5

        if ov > 5:
            Kd *= 2.0

        Kd = np.clip(Kd, 0.0001, 500)

        return [float(Kd)]

    # ====================================
    # Problem b
    # ====================================
    elif problem == 'b':

        Kp, Ki = last_params

        tr = float(last_metrics.get('tr', 1))
        ov = float(last_metrics.get('overshoot', 0))
        ess = float(last_metrics.get('e_ss_acc', 1))

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

        if bw < 1000:
            Kp *= 1.15
            Kd *= 1.10

        if pm < 70:
            Kd *= 1.25

        if mr > 1.1:
            Kd *= 1.20
            Kp *= 0.92

        if ess > 0.2:
            Ki *= 1.20

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

    try:

        system_prompt = f"""
你是 PID 控制专家。

问题类型: {problem}

规格要求:
{get_specs_text(problem)}

请输出下一组参数。

只能输出 JSON。
"""

        user_msg = ""

        for h in history:

            user_msg += f"""
迭代 {h['iteration']}
参数: {h['params']}
指标: {h['metrics']}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=100
        )

        reply = response.choices[0].message.content.strip()

        start = reply.find("{")
        end = reply.rfind("}")

        if start != -1 and end != -1:
            reply = reply[start:end+1]

        data = json.loads(reply)

        if problem == 'a':

            return [
                float(data.get("Kd", 0.05))
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

    if problem == 'a':
        params = [0.05]

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

        print("\n============================")
        print(f"迭代 {i+1}")
        print("============================")

        print("参数:", params)

        print("指标:")

        for k, v in metrics.items():
            print(f"{k}: {v}")

        print("达标:", satisfied)

        if satisfied:

            best_params = params.copy()
            best_metrics = metrics
            best_satisfied = True

            break

        if best_metrics is None:
            best_metrics = metrics
            best_params = params.copy()

        params = ask_llm_for_params(problem, history)

        params = [max(0.0, float(p)) for p in params]

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
