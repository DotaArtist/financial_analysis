#!/user/bin/env python3
# -*- coding: utf-8 -*-
# 税率计算
# Created by yewen.yp

# 养老保险单位缴纳比例：21%;个人缴纳比例：8%
# 医疗保险单位缴纳比例：11%;个人缴纳比例：2%
# 失业保险单位缴纳比例：1.5%;个人缴纳比例：0.5%
# 生育保险单位缴纳比例：1%;个人不承担缴费
# 工伤保险单位缴纳比例：0.5%;个人不承担缴费
# 公积金单位缴纳比例：24%;个人缴纳：12%

# 累计应纳税额=累计税前收入-累计免征额（免征额为每月5000元）-累计社保公积金个人部分-累计专项附加扣除金额-个人养老金免税额；
# 应缴纳个税金额=累计应纳税额*对应税档-速算扣除数-已扣缴个税。

# 1. 应纳税所得额不超过36,000元的部分，税率为3%，速算扣除数为0元。
# 2. 应纳税所得额超过36,000元至144,000元的部分，税率为10%，速算扣除数为2,520元。
# 3. 应纳税所得额超过144,000元至300,000元的部分，税率为20%，速算扣除数为16,920元。
# 4. 应纳税所得额超过300,000元至420,000元的部分，税率为25%，速算扣除数为31,920元。
# 5. 应纳税所得额超过420,000元至660,000元的部分，税率为30%，速算扣除数为52,920元。
# 6. 应纳税所得额超过660,000元至960,000元的部分，税率为35%，速算扣除数为85,920元。
# 7. 应纳税所得额超过960,000元的部分，税率为45%，速算扣除数为181,920元。

# 奖金单独计税和合并计税
# 例如，老李全年工资20万元，年底奖金2.4万元，假设可享受三险一金、赡养老人等扣除共4.4万元。在年度汇算时，他该怎么缴税呢?
# 一是将2.4万元奖金和20万元工资合并计税，扣除6万元（每月5000元）减除费用和4.4万元后，得到应纳税所得额12万元，按照综合所得年度税率表，应纳税120000×10%-2520=9480元；
# 二是将2.4万元奖金单独计税，年底奖金应纳税720元，20万元工资扣除6万元减除费用和4.4万元后，应纳税96000×10%-2520=7080元，合计应纳税7800元。两种方式下，税额相差1680元，老李选择单独计税更划算。


def function_rate(tax_money):
    """最新税率"""
    rate, speed_money = 0., 0
    if tax_money <= 36000:
        rate, speed_money = 0.03, 0
    elif 36000 < tax_money <= 144000:
        rate, speed_money = 0.10, 2520
    elif 144000 < tax_money <= 300000:
        rate, speed_money = 0.20, 16920
    elif 300000 < tax_money <= 420000:
        rate, speed_money = 0.25, 31920
    elif 420000 < tax_money <= 660000:
        rate, speed_money = 0.30, 52920
    elif 660000 < tax_money <= 960000:
        rate, speed_money = 0.35, 85920
    elif tax_money > 960000:
        rate, speed_money = 0.45, 181920
    return rate, speed_money


def work_money(salary, house_rate=7, special_additional_deductions=0, current_month=1, bonus=0, all_rate=1., bonus_type="单独", house_base=36921):
    """
    :param salary: str or list
    :param house_rate: 住房公积金
    :param special_additional_deductions: 附加扣除
    :param current_month: 当前月份
    :param bonus: 奖金
    :param all_rate: 社保缴纳比例
    :param bonus_type: 单独 or 合并
    :param house_base
    :return:
    """
    if isinstance(salary, list):
        salary_current = salary[0]
    else:
        salary_current = salary
    cum_money_out_pre, cum_money_out, cum_money = 0, 0, 0  # 历史已缴纳税, 当月应缴税, 累计应纳税额
    house_money = house_base if salary_current >= house_base else salary_current
    house_tax = all_rate * house_money * house_rate * 0.01
    social_security = all_rate * salary_current * (8 + 2 + 0.5 + 0 + 0) * 0.01 + house_tax
    for i in range(current_month):
        if isinstance(salary, list):
            salary_current = salary[i]
        else:
            salary_current = salary
        house_money = house_base if salary_current >= house_base else salary_current
        house_tax = all_rate * house_money * house_rate * 0.01
        social_security = all_rate * salary_current * (8 + 2 + 0.5 + 0 + 0) * 0.01 + house_tax
        if bonus_type == "合并":
            cum_money += (salary_current + bonus / 12 - 5000 - social_security - special_additional_deductions)  # 累计应纳税额
        else:
            cum_money += (salary_current - 5000 - social_security - special_additional_deductions)  # 累计应纳税额
        cum_rate, cum_speed_money = function_rate(cum_money)
        cum_money_out = (cum_money * cum_rate - cum_speed_money - cum_money_out_pre)  # 当月应缴税
        cum_money_out_pre += cum_money_out  # 累计已缴纳个税

    if bonus_type == "单独":
        cum_rate, cum_speed_money = function_rate(bonus)
        cum_money_out_pre += bonus * cum_rate - cum_speed_money
    print("月份：", current_month, "当月应缴纳个税:", round(cum_money_out, 2), "当月五险一金:", round(social_security, 2),
          "当月到手:", round(salary_current - cum_money_out - social_security, 2), "累计缴税:", round(cum_money_out_pre, 2))
    return salary_current - social_security - cum_money_out, cum_money_out_pre


def cdf_work_money(salary, house_rate=7, special_additional_deductions=0, current_month=9, bonus=0, all_rate=1., bonus_type="单独", house_base=36921):
    out = 0
    for i in range(current_month):
        tmp, _ = work_money(salary, house_rate, special_additional_deductions, i+1, bonus, all_rate, bonus_type, house_base)
        out += tmp
    print(round(out + bonus, 2))


if __name__ == '__main__':
    cdf_work_money(salary=1000, house_rate=7, special_additional_deductions=2000, current_month=12, bonus=52000*2, all_rate=1, bonus_type="单独")
