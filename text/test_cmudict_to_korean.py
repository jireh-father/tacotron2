from text.cmuToKorean import CMUToKorean
import cmudict # dup in this dir

ret = CMUToKorean.convert('seventeen', " ".join(cmudict.dict()["seventeen"][0]))
print(ret)