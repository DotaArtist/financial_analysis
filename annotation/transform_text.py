#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'


def transform_text(str_a):
    out = []
    counter = 1
    for i in str_a:
        # if counter % 100 == 0:
        #     out.append(" ")

        out.append(i)
        counter += 1
    return " ".join(out)


a = """克,临邑县人民医院,姓名:马梓琪,出院记录,住院号:455231,姓名:马梓琪,入院日期:2019/2/421:16:59,性别:女,出院日期:2019/2/815:08:11,年龄:2岁3月,住院天数:4天,入院情况:患儿因“发热、腹痛、呕吐3天,声音嘶哑、尿少1天”入院,热峰,1次/日,峰值39.1℃,发热时伴有阵发性腹痛,脐周为主,呕吐1次/日,声音嘶哑,无喉,鸣,无咳喘,无憋气,尿量减少,色发黄,院外治疗,效果欠佳,来院。体检:T39℃,P126次/分,R30次/分,Bp-/-mmhg,wt10kg精神欠佳;全身皮肤黏膜弹性欠佳,略干燥,口唇略干燥,咽部充血,颈软,双肺呼吸音粗,未闻及罗音;心律齐,心音有力,各瓣膜听,诊区未闻及杂音;腹软,无压痛及反跳痛,未触及包块,肠鸣音正常。辅助检查:腹部超声,结果示脐周及右中下腹部肠系膜走行区探及多个,实性低回声结节,较大者1.4*0.7cm,边,界清,内回声均匀。提示肠系膜根部淋巴结炎;血常规示WBC8.76*10/L,NEU%63.5%,lym%23.2%,RbC4.73*1012/L,HgB13g/L;PLT293*10/L,超敏CRP11.36mg/L;心肌酶示,AST38U/L,CK54U/L,CK-MB45U/L,LDH63U/L,HBDH193U/L;离子四项钾离子4.5mmol/L,钠离子134mmol/L,氯离子101mmol/L,二氧化碳结合力18mmol/L,入院诊断:1.急性喉炎,2.心肌损害,3.急性肠系膜淋巴结炎并轻度脱水,4.代谢性酸中,毒,诊疗经过:入院后给予儿科护理常规,一级护理,患儿家属拒绝为患儿抽血化验肝功,能、支原体及衣原体抗体、抗“0,血培养及胸片检查并签字。给予口服牛磺酸颗粒清热,解毒,磷酸奥司他韦颗粒抗感染,雾化吸入布地奈德抗炎,静脉输注甲强龙抗炎,西咪替丁,保护胃黏膜,小苏打纠酸,热毒宁、头孢美唑钠抗感染,磷酸肌酸钠营养心肌细胞等综合治,疗。患儿病情逐渐好转。2019-02-06复查血常规示白细胞:7.30*10^9/L;中性粒细胞比率:,3.80%;淋巴细胞比率:53.60%;红细胞:3.9*10^12/L;血红蛋白:107.00g/L;血小,板:297.00*10^9/L;超敏C反应蛋白:1.31mg/L,出院诊断:1.急性喉炎,2.心肌损害,3.急性肠系膜淋巴结炎并轻度脱水,4.代谢性酸中,毒,出院情况:患儿无腹痛,无呕吐,无发热,无声音嘶哑,饮食睡眠佳,二便正常。查,体:体温正常,精神佳,呼吸平稳,咽部无充血;颈软,双肺呼吸音粗,未闻及罗音;心律,齐,心音有力;肝脾未及肿大,腹软,无压痛,未触包块,肠鸣音正常;神经系统未查及,第1页"""
print(transform_text(a))