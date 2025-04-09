from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

device='cuda'
model.to(device)

# 设置生成参数
def generate_with_cot(prompt, max_length=1000):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs["attention_mask"] = inputs.input_ids.ne(tokenizer.pad_token_id).int()
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=0.6, 
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

problem = "解方程组：\n方程1: x + y = 8\n方程2: 2x - y = 1"

# CoT提示：显式要求分步推理
cot_prompt = f"""
请逐步解决以下问题：

{problem}

分步推理：
1. 观察方程组的结构，寻找消元方法。
2. 通过相加方程消去变量y。
3. 解出x的值。
4. 代入求y。
5. 验证解的正确性。

"""

response = generate_with_cot(cot_prompt)
print(response)

def plot_equations_solution(x_sol, y_sol):
    plt.figure(figsize=(8, 6))
    
    # 生成x值范围
    x = np.linspace(0, 10, 100)
    
    # 绘制方程1: x + y = 8 → y = 8 - x
    y1 = 8 - x
    plt.plot(x, y1, label='equation1: x + y = 8')
    
    # 绘制方程2: 2x - y = 1 → y = 2x - 1
    y2 = 2*x - 1
    plt.plot(x, y2, label='equation2: 2x - y = 1')
    
    # 标出解点
    plt.scatter(x_sol, y_sol, c='red', zorder=5, 
               label=f'Solution ({x_sol}, {y_sol})')
    
    plt.title("Visualization of systems of equations")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.axis([0, 10, 0, 10])
    plt.show()

# 解析函数
def parse_solution(response):
    # 正则表达式匹配最终解
    solution_pattern = re.compile(
        r'(?:解为|答案|最终解)[^\d]*?'  # 匹配解的前导关键词
        r'x\s*[=＝]\s*([+-]?\d+\.?\d*)'  # 匹配x值，支持=和＝符号
        r'[^\d]*?'  # 非数字分隔符
        r'y\s*[=＝]\s*([+-]?\d+\.?\d*)',  # 匹配y值
        re.IGNORECASE | re.DOTALL
    )
    
    match = solution_pattern.search(response)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        # 备用匹配：全局搜索最后一次出现的x=和y=
        x_matches = list(re.finditer(r'x\s*[=＝]\s*([+-]?\d+\.?\d*)', response))
        y_matches = list(re.finditer(r'y\s*[=＝]\s*([+-]?\d+\.?\d*)', response))
        if x_matches and y_matches:
            return float(x_matches[-1].group(1)), float(y_matches[-1].group(1))
        print("解析失败，请确保模型输出包含类似'x=3, y=5'的明确解")
        return None, None

# 新增可视化处理
x_sol, y_sol = parse_solution(response)
if x_sol is not None and y_sol is not None:
    plot_equations_solution(x_sol, y_sol)
else:
    print("无法生成可视化，解析解失败")