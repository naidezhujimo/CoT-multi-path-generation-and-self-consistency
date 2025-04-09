from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
from sympy import symbols, Eq, solve, sympify


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_map="auto"
)

# CoT
def generate_cot_prompt(problem, method=None):
    method_hints = {
        "algebra": [
            "观察方程结构，寻找可消元变量",
            "通过加减方程消去一个未知数",
            "解出单一变量后回代求解",
            "验证解是否满足所有方程"
        ],
        "matrix": [
            "将方程组写成矩阵形式AX=B",
            "计算系数矩阵A的行列式",
            "若行列式不为零，计算逆矩阵A⁻¹",
            "通过X=A⁻¹B求解未知数"
        ],
        "default": [
            "分析方程组的线性组合可能性",
            "选择最简便的消元方法",
            "逐步推导并记录中间结果",
            "交叉验证所有方程"
        ]
    }

    selected_method = method if method in method_hints else "default"
    steps = "\n".join([f"{i+1}. {hint}" for i, hint in enumerate(method_hints[selected_method])])
    
    return f"""请严格按以下格式回答：

{problem}

分步思考（{selected_method}方法）：
{steps}

必须使用XML标签包裹答案：
<solution>
[此处写推导过程，必须包含具体计算步骤]
</solution>
<final_answer>
[此处写最终答案，格式示例：x=3,y=5]
</final_answer>"""

# 多路径生成
def multi_path_generation(problem, num_paths=3):
    methods = ["algebra", "matrix", "default"]
    paths = []
    
    for i in range(num_paths):
        method = methods[i % len(methods)]
        prompt = generate_cot_prompt(problem, method)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7 + 0.15*i,
            top_p=0.92,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "</final_answer>" in full_text:
            paths.append(full_text)
    
    return paths

# 自洽性验证器
def self_consistency_check(paths):
    answer_pattern = r"<final_answer>([\s\S]*?)</final_answer>"
    answer_counter = Counter()
    
    for path in paths:
        match = re.search(answer_pattern, path, re.IGNORECASE)
        if match:
            raw_answer = match.group(1)
            # 标准化处理
            cleaned = re.sub(r"[^0-9a-zA-Z=,]", "", raw_answer)
            cleaned = re.sub(r"([a-zA-Z])=", r"\1=", cleaned)
            answer_counter[cleaned.lower()] += 1
    
    # 多数投票需超过60%
    if answer_counter:
        most_common = answer_counter.most_common(1)[0]
        return most_common[0] if most_common[1] > len(paths)*0.6 else None
    return None

# 数学验证层
def mathematical_verification(problem, answer):
    try:
        problem_clean = re.sub(r"[\u2010-\u2015\-]", "-", problem)
        problem_clean = re.sub(r"解方程组：|方程\d+[:：]\s*", ";", problem_clean)
        problem_clean = re.sub(r"[^0-9xyab=+\-*/();]", " ", problem_clean)
        problem_clean = re.sub(r"(\d)([xyab])", r"\1*\2", problem_clean)
        problem_clean = re.sub(r"\s+", " ", problem_clean).strip()
        problem_clean = re.sub(r";+", ";", problem_clean).strip(';')

        equations = []
        equation_strings = [eq.strip() for eq in problem_clean.split(';') if eq.strip()]
        
        for eq_str in equation_strings:
            match = re.match(r"^\s*([^=]+?)\s*=\s*(.+?)\s*$", eq_str)
            if not match:
                print(f"方程格式错误: {eq_str}")
                return False
            lhs, rhs = match.groups()
            try:
                equations.append(Eq(sympify(lhs), sympify(rhs)))
            except Exception as e:
                print(f"方程解析失败: {lhs} = {rhs}，错误: {e}")
                return False
        
        answer_dict = {}
        for pair in answer.split(','):
            var, val = pair.split('=')
            var = var.strip().lower()
            answer_dict[symbols(var)] = float(val)
        
        return all(eq.subs(answer_dict) for eq in equations)
    except Exception as e:
        print(f"验证错误: {str(e)}")
        return False

from sympy import lambdify

def visualize_equations(problem, solution=None):
    """绘制方程组图形"""
    try:
        # 解析方程
        x, y = symbols('x y')
        eqs = []
        problem_clean = re.sub(r"[\u2010-\u2015\-]", "-", problem)
        problem_clean = re.sub(r"解方程组：|方程\d+[:：]\s*", ";", problem_clean)
        problem_clean = re.sub(r"[^0-9xyab=+\-*/();]", " ", problem_clean)
        problem_clean = re.sub(r"(\d)([xyab])", r"\1*\2", problem_clean)
        problem_clean = re.sub(r"\s+", " ", problem_clean).strip()
        problem_clean = re.sub(r";+", ";", problem_clean).strip(';')
        # 分割方程并解析
        equation_strings = [eq.strip() for eq in problem_clean.split(';') if eq.strip()]
        for eq_str in equation_strings:
            match = re.match(r"^\s*([^=]+?)\s*=\s*(.+?)\s*$", eq_str)
            if not match:
                print(f"方程格式错误: {eq_str}")
                return False
            lhs, rhs = match.groups()
            try:
                eqs.append(Eq(sympify(lhs), sympify(rhs)))
            except Exception as e:
                print(f"方程解析失败: {lhs} = {rhs}，错误: {e}")
                return False

        # 生成坐标数据
        x_vals = np.linspace(-10, 10, 400)
        plt.figure(figsize=(8,6))
        colors = ['#1f77b4', '#ff7f0e']  # 不同方程的颜色
        
        for i, eq in enumerate(eqs):
            try:
                # 解方程获取y关于x的表达式
                y_expr = solve(eq, y)[0]
                y_func = lambdify(x, y_expr, modules='numpy')
                y_vals = y_func(x_vals)
                plt.plot(x_vals, y_vals, label=f'equation {i+1}', color=colors[i], lw=2)
            except:
                # 处理无法解析为y=f(x)的情况
                x_expr = solve(eq, x)[0]
                x_func = lambdify(y, x_expr, modules='numpy')
                y_vals = np.linspace(-10, 10, 400)
                x_vals = x_func(y_vals)
                plt.plot(x_vals, y_vals, label=f'equation {i+1}', color=colors[i], lw=2)

        # 标出解点
        if solution:
            x_val = float(solution.split('x=')[1].split(',')[0])
            y_val = float(solution.split('y=')[1])
            plt.scatter(x_val, y_val, color='red', s=100, 
                        zorder=5, label=f'Solution ({x_val}, {y_val})')
            plt.annotate(f'({x_val}, {y_val})', (x_val+0.5, y_val),
                         fontsize=10, color='darkred')

        plt.title("Visualization of systems of equations")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        # 保存图片或直接显示
        plt.savefig('equation_visualization.png', dpi=300)
        plt.close()
        print("可视化结果已保存至 equation_visualization.png")
        
    except Exception as e:
        print(f"可视化失败: {str(e)}")


# 工作流程
def full_pipeline(problem):
    # 预处理输入问题
    problem = re.sub(r"\s+", " ", problem)  # 合并多余空格
    problem = problem.replace("−", "-")     # 统一减号

    print(f"\n{'='*40}\n处理问题: {problem}\n{'='*40}")
    
    # 生成多路径
    paths = multi_path_generation(problem, num_paths=5)
    print(f"\n生成有效路径数: {len(paths)}/{5}")
    
    # 打印示例路径
    for i, path in enumerate(paths[:2]):
        print(f"\n示例路径 {i+1}:\n{'-'*30}")
        print(re.sub(r"(<solution>.*?</final_answer>)", r"\1", path, flags=re.DOTALL))
    
    # 自洽性验证
    final_answer = self_consistency_check(paths)
    print(f"\n自洽性选择结果: {final_answer or '无一致答案'}")
    
    # 数学验证
    if final_answer:
        is_valid = mathematical_verification(problem, final_answer)
        print(f"数学验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")
        if is_valid:
            visualize_equations(problem, final_answer)
            return f"\n最终结果: 验证通过 ✅\n答案: {final_answer}"
    
    # 失败处理
    error_info = "\n失败分析:\n"
    if not paths:
        error_info += "- 所有路径生成失败，请检查模型加载或提示工程"
    elif not final_answer:
        error_info += "- 路径间答案不一致，建议:\n  1. 增加生成路径数\n  2. 优化提示模板"
    else:
        error_info += "- 数学验证未通过，建议:\n  1. 检查方程解析逻辑\n  2. 添加更多验证约束"
    
    return error_info

# 运行
if __name__ == "__main__":
    test_cases = [
        "解方程组：\n方程1: x + y = 8\n方程2: 2x - y = 1",
    ]
    
    for problem in test_cases:
        result = full_pipeline(problem)
        print(result)
        print("\n" + "="*60)