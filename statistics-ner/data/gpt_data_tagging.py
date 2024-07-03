import openai
import csv
import time
import pandas as pd

explame1 = '''本文用相关分析说明了〈中学生素质特点分类培养〉项目研究中所用各种心理诊断量表整体组合运用的合理性 ;
用因子分析简化了测试项目的指标体系 ,选定了适当的公共主因子 ,并对公因子给予了合理的解释 ;
用聚类分析依据因子得分对学生实施心理素质特点分类 ;根据学生的心理素质特点类型提出了相应的宏观培养策略'''

explame2 = '''文章研究了纵向数据非参数模型y=f(t)+ε,其中f(t)为未知平滑函数,ε为零均值随机误差项。
我们选取一组基函数对f(t)进行基函数展开近似,然后构造关于基函数系数的二次推断函数,利用New ton-Raphson迭代方法得到基函数系数的估计值,进而得到未知平滑函数f(t)的拟合估计。
理论结果显示,所得到的基函数系数估计有相合性和渐近正态性。最后通过数值方法得到了较好的模拟结果。'''

explame3 = '''文章以湖北省16个城市的金融资源为研究对象,基于DEA模型测度其金融资源投入产出效率,通过效率的空间关联、空间差异以及空间转移的定量分析,验证了湖北省金融资源的非均衡分布的特征。
结果显示,湖北省金融资源投入产出效率呈现稳步发展态势,且通过空间莫兰指数发现其在经济距离上存在集聚的正相关特征,湖北省各市的金融资源空间分布差距较大的问题并未得到缓解。
通过空间马尔科夫链模型发现,湖北金融资源发展呈现"高水平垄断"与"低水平陷阱"的特征,各市金融资源发展的空间非均衡容易受到相邻地区的影响,形成遇强则强、遇弱则弱的马太效应。因此在缓解各市金融非均衡发展的问题上,需要有空间视角的考量,通过空间互动、空间关联的方式有效配置地区间金融资源。'''

sys_prompt = f'''你是一位专业的统计学命名实体识别专家。负责从统计学相关论文的摘要中提取论文的研究对象和研究方法。
你需要识别论文摘要中提及的研究对象和研究方法，并仿照以下格式输出：
[研究对象];[研究方法1,研究方法2]
###
你需要参考以下例子进行学习：
摘要：{explame1}
结果：[中学生素质特点];[相关分析,因子分析,聚类分析]
###
如果摘要中没有提及研究对象或研究方法，则返回[];[]
你只需要给出结果，而无需返回推理过程。'''


openai.api_type = "your_api_type"
openai.api_base = "your_api_base"
openai.api_key = 'your_api_key'
openai.api_version = "your_api_version"
engine = "your_gpt_engine"

def save_to_csv(id, save_path, response:str, abstract:str):
    with open(save_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows([[id]+[abstract]+response.split(';')])

def gpt(abstract:str, temperature:float=0.1, max_tokens:int=4096, top_p:float=0.1)->str:
    response = openai.ChatCompletion.create(
        engine = engine,
        messages=[{"role": "system", "content": sys_prompt},
                  {'role': 'user', 'content': '摘要:'+explame2},
                  {'role': 'assistant', 'content':'[纵向数据非参数模型];[New ton-Raphson迭代]'},
                  {'role': 'user', 'content': '摘要:'+explame3},
                  {'role': 'assistant', 'content':'[金融资源];[DEA模型,定量分析,空间莫兰指数,空间马尔科夫链模型]'},
                  {'role': 'user', 'content': '摘要:'+abstract}],
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = top_p,
        frequency_penalty = 0,
        presence_penalty = 0,
        stop = None)
    answer = response["choices"][0].message.content
    return answer