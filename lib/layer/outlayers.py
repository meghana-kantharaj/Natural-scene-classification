import numpy as np
import theano as th
import theano.tensor as tt
from .hidden import HiddenLayer
from .weights import borrow, is_shared_var

float_x = th.config.floatX
############################### Output Layer  ##################################


class OutputLayer(object):
    def cost(self, y):
        if self.loss == "nll":
            return self.neg_log_likli(y)

        elif self.loss == "nllsq":
            return self.neg_log_likli_sq(y)

        elif self.loss.startswith("nll"):
            try:
                threshold = int(self.loss[-2:])/100
                threshold = np.clip(threshold, 0, 1)
            except ValueError:
                print("Did not understand {}, using plain NLL".format(self.loss))
                threshold = 1.0

            return self.neg_log_likli_trunc(y, threshold)

        elif self.loss == "hinge":
            return self.hinge(y)

        else:
            raise NotImplementedError("Loss : " + self.loss)

    def neg_log_likli_sq(self, y):
        return tt.mean(self.logprob[tt.arange(y.shape[0]), y]**2)

    def neg_log_likli_trunc(self, y, threshold):
        print("Using threshold: ", threshold)
        logthreshold = np.log(threshold)    # A negative number
        return tt.mean(tt.maximum(0, logthreshold
                                  -self.logprob[tt.arange(y.shape[0]), y]))

    def neg_log_likli(self, y):
        return -tt.mean(self.logprob[tt.arange(y.shape[0]), y])

    def hinge_max(self, y):
        print("Using Hinge Loss!!!")
        def step(out, y_):
            return tt.maximum(0, 1 +
              tt.max(tt.concatenate((out[:y_],out[y_+1:self.n_out]))) - out[y_])

        losses, _ = th.scan(step, sequences=[self.output, y])
        return tt.mean(losses)

    def hinge(self, y):
        return tt.mean(tt.maximum(0, self.output + 1 -
                self.output[tt.arange(y.shape[0]), y].dimshuffle(0, 'x')))

    def features_and_predictions(self,y):
        return self.features, self.y_preds,y

    def sym_and_oth_err_rate(self, y):
        sym_err_rate = tt.mean(tt.neq(self.y_preds, y))

        if self.kind == 'LOGIT':
            # Bit error rate
            second_stat = tt.mean(self.bitprob[tt.arange(y.shape[0]), y] < .5)

        else:
            # Likelihood of MLE
            second_stat = tt.mean(self.probs[tt.arange(y.shape[0]), y])

        return sym_err_rate, second_stat



class SoftmaxLayer(HiddenLayer, OutputLayer):
    def __init__(self, inpt, wts, rand_gen=None, n_in=None, n_out=None,
                 reg=(),
                 loss="nll"):
        HiddenLayer.__init__(self, inpt, wts, rand_gen, n_in, n_out,
                             actvn='Softmax', reg=reg,
                             pdrop=0)
        self.y_preds = tt.argmax(self.output, axis=1)
        self.probs = self.output
        self.logprob = tt.log(self.probs)
        self.features = self.logprob
        self.kind = 'SOFTMAX'
        self.loss = loss
        self.representation = "Softmax In:{:3d} Out:{:3d} Loss:{}" \
            "\n\t  L1:{L1} L2:{L2} Momentum:{momentum} Max Norm:{maxnorm} " \
            "Rate:{rate}""".format(self.n_in, self.n_out,
                                   self.loss, **self.reg)

    def TestVersion(self, inpt):
        return SoftmaxLayer(inpt, (self.w, self.b))
