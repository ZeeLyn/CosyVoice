from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph
from functools import partial

print(split_paragraph("东北雨姐是真的塌房了。被央视爸爸三次点名。视频里骑的不是自行车。就是电动车。要么就是小三轮。小货车的东北雨姐。团队成员都开什么车呢？其实啊背后个个都是豪车。尤其是最后一位。让我都惊掉下巴。首先。雨姐本人今年刚入手一辆奔驰S500。落地200多万。在团队里充当伙夫的火神佩斯。座驾是一辆宝马。雨姐家女团的大翠花。去年新购入一辆奔驰E300 。就连起早贪黑做豆腐的大华。开的都是60多万的酷路泽。而大乔开的是奥迪A6。同时还有一辆劳斯莱斯库里南。好家伙。果然淳朴的农村生活都是演给家人们看的。", partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                         token_min_n=60, merge_len=20, comma_split=False))