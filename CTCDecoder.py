import torch
import beamsearch
import beamsearchcode128


code128 = {
    '212222':0,
    '222122':1,
    '222221':2,
    '121223':3,
    '121322':4,
    '131222':5,
    '122213':6,
    '122312':7,
    '132212':8,
    '221213':9,
    '221312':10,
    '231212':11,
    '112232':12,
    '122132':13,
    '122231':14,
    '113222':15,
    '123122':16,
    '123221':17,
    '223211':18,
    '221132':19,
    '221231':20,
    '213212':21,
    '223112':22,
    '312131':23,
    '311222':24,
    '321122':25,
    '321221':26,
    '312212':27,
    '322112':28,
    '322211':29,
    '212123':30,
    '212321':31,
    '232121':32,
    '111323':33,
    '131123':34,
    '131321':35,
    '112313':36,
    '132113':37,
    '132311':38,
    '211313':39,
    '231113':40,
    '231311':41,
    '112133':42,
    '112331':43,
    '132131':44,
    '113123':45,
    '113321':46,
    '133121':47,
    '313121':48,
    '211331':49,
    '231131':50,
    '213113':51,
    '213311':52,
    '213131':53,
    '311123':54,
    '311321':55,
    '331121':56,
    '312113':57,
    '312311':58,
    '332111':59,
    '314111':60,
    '221411':61,
    '431111':62,
    '111224':63,
    '111422':64,
    '121124':65,
    '121421':66,
    '141122':67,
    '141221':68,
    '112214':69,
    '112412':70,
    '122114':71,
    '122411':72,
    '142112':73,
    '142211':74,
    '241211':75,
    '221114':76,
    '413111':77,
    '241112':78,
    '134111':79,
    '111242':80,
    '121142':81,
    '121241':82,
    '114212':83,
    '124112':84,
    '124211':85,
    '411212':86,
    '421112':87,
    '421211':88,
    '212141':89,
    '214121':90,
    '412121':91,
    '111143':92,
    '111341':93,
    '131141':94,
    '114113':95,
    '114311':96,
    '411113':97,
    '411311':98,
    '113141':99,
    '114131':100,
    '311141':101,
    '411131':102,
    '211412':103,
    '211214':104,
    '211232':105,
    '2331112':106
}

def decode(code):
    """
    code to codevalue
    yes: codevalue
    no : []
    """
    str_code = list(map(lambda x:str(x), code))
    length = len(str_code)
    codevalue = []
    if (length - 7) % 6 or (length - 7) / 6 < 2:
        return []
    for start in range(0, length - 7, 6):
        meta = ''.join(str_code[start:start+6])
        if meta not in code128:
            return []
        codevalue.append(code128[meta])
    if codevalue[0] < 103 or ''.join(str_code[-7:]) not in code128:
        return []
    codevalue.append(code128[''.join(str_code[-7:])])
    valid_sum = sum([index*value for index, value in enumerate(codevalue[:-2])]) + codevalue[0]
    if valid_sum % 103 != codevalue[-2]:
        return []
    return codevalue


def remove_blank(labels, blank=0):
    """
    输入:labels [T]
    输出:new_labels [S]
    """
    new_labels = []
    previous = None
    for label in labels:
        if label != previous:
            new_labels.append(label)
            previous = label
    new_labels = [int(label) for label in new_labels if label != blank]
    return torch.Tensor(new_labels)

class CTCDecoder:
    def __init__(self, beam_size=100, num_processes=8, prob_length=450):
        self.beam_size = beam_size
        self.num_processes = num_processes
        self.prob_length = prob_length

    def Greedy(self, probs):
        """
        input: [B,T,C] log_softmax
        output:
            codes:[B,T]{list(tensor)}
            values:[B,T]{list(list)}
            scores:[B]{list}
        """
        preds_value, preds_index = probs.topk(1, 2, True, True)

        codes = []
        values = []
        scores = []
        for pred_index, pred_value in zip(preds_index, preds_value):
            code = remove_blank(pred_index).int()
            value = decode(code.tolist())
            score = sum(pred_value).exp().item()
            if value == []:
                code = torch.Tensor([]).int()
            codes.append(code)
            values.append(value)
            scores.append(score)
        return codes, values, scores


    def BeamSearch(self, probs):
        """
        input: [B,T,C] log_softmax
        output:
            codes:[B,T]{list(tensor)}
            values:[B,T]{list(list)}
            scores:[B]{list}
        """
        B, T, C = probs.shape
        assert len(probs.shape) == 3 and T == self.prob_length and C == 5, "error: The shape of Tensor is need [B x " + str(self.prob_length) + " x 5] but get " + str(len(probs.shape))
        
        output = torch.full((B, self.beam_size, T), 0.0).cpu().int()
        timesteps = torch.full((B, self.beam_size, T), 0.0).cpu().int()
        codeValues = torch.full((B, self.beam_size, T), 0.0).cpu().int()
        scores = torch.FloatTensor(B, self.beam_size).cpu().float()
        out_length = torch.zeros(B, self.beam_size).cpu().int()

        beamsearch.paddle_beam_decode(probs.cpu(), torch.IntTensor([self.prob_length]*B).int().cpu(), self.beam_size,
                                      self.num_processes,
                                      0, output, timesteps,
                                      codeValues, scores, out_length)

        codes = []
        values = []
        scores_exp = []
        for pred_index, pred_score, pred_length in zip(output, scores, out_length):
            # find res from beamsearch result(len = beamsize)
            find = False
            for index, length, score in zip(pred_index, pred_length, pred_score):
                code = index[:length].int()
                value = decode(code.tolist())
                if value != []:
                    score_exp = score.exp().item()
                    find = True
                    break
            if not find:
                code = torch.Tensor([]).int()
                value = []
                score_exp = 0.0
            codes.append(code)
            values.append(value)
            scores_exp.append(score_exp)
        return codes, values, scores_exp
    

    def BeamSearchCode128(self, probs):
        """
        input: [B,T,C] log_softmax
        output:
            codes:[B,T]{list(tensor)}
            values:[B,T]{list(list)}
            scores:[B]{list}
        """
        B, T, C = probs.shape
        assert len(probs.shape) == 3 and T == self.prob_length and C == 5, "error: The shape of Tensor is need [B x " + str(self.prob_length) + " x 5] but get " + str(len(probs.shape))
        
        output = torch.full((B, self.beam_size, T), 0.0).cpu().int()
        timesteps = torch.full((B, self.beam_size, T), 0.0).cpu().int()
        codeValues = torch.full((B, self.beam_size, T), 0.0).cpu().int()
        scores = torch.FloatTensor(B, self.beam_size).cpu().float()
        out_length = torch.zeros(B, self.beam_size).cpu().int()
        
        beamsearchcode128.paddle_beam_decode(probs.cpu(), torch.IntTensor([self.prob_length]*B).int().cpu(), self.beam_size,
                                             self.num_processes,
                                             0, output, timesteps,
                                             codeValues, scores, out_length)
        codes = []
        values = []
        scores_exp = []
        for pred_index, pred_value, pred_score, pred_length in zip(output, codeValues, scores, out_length):
            code = pred_index[0][:pred_length[0]].int()
            value = pred_value[0][:pred_length[0]//6].tolist()
            score_exp = pred_score[0].exp().item()
            codes.append(code)
            values.append(value)
            scores_exp.append(score_exp)
        return codes, values, scores_exp

    
