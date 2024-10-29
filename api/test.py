import os,io
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/..'.format(ROOT_DIR))
sys.path.append('{}/../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice_plus import CosyVoicePlus
cosyvoice = CosyVoicePlus('.\\pretrained_models\\CosyVoice-300M\\')
texts=cosyvoice.text_normalize('''哥哥们，今天小师妹不聊车，聊车企，我先说三个点，如果有能猜对的哥哥，知道我说的是哪家车企，小师妹会送出我的神秘惊喜。第一、自身优势是发动机技术。第二、出口第一，得到大量海外用户认可。第三、国产车里最低调的车企
小师妹用一条视频，告诉你我怎么看。
先看看国产新能源9月份销量前十的数据。这些汽车品牌中有稳居前茅的，也有突飞猛进的，而能飞速增长的，那必然依靠的是技术支撑。
今年这个企业在行业内，不管是燃油车还是新能源，国内海外都是双增长，对比其他销量下滑，甚至在国内各种减产的汽车品牌，今年前三季度累计销量175万辆，同比增长了39.9%，还进了世界500强第385位
未来不管是燃油增程还是混动，都离不开发动机，再加上一生都要强的中国人，会不断创新，一个品牌，让未来的中国汽车市场就站稳了脚跟，凭着自己的技术，只会让国产的汽车更强
通过上面的各种描述，哥哥们猜到我说的是哪家车企了嘛？''')

print(texts)
