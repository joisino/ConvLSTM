import numpy as np
import chainer
from chainer import Variable
from chainer import variable
from chainer import reporter
from chainer import initializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
from PIL import Image

class ConvLSTM(Chain):
    def __init__(self, inp = 256, mid = 128, sz = 3):
        super(ConvLSTM, self).__init__(
            Wxi = L.Convolution2D(inp, mid, sz, pad = sz//2),
            Whi = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True),
            Wxf = L.Convolution2D(inp, mid, sz, pad = sz//2),
            Whf = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True),
            Wxc = L.Convolution2D(inp, mid, sz, pad = sz//2),
            Whc = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True),
            Wxo = L.Convolution2D(inp, mid, sz, pad = sz//2),
            Who = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True)
        )

        self.inp = inp
        self.mid = mid
        
        self.pc = None
        self.ph = None

        with self.init_scope():
            Wci_initializer = initializers.Zero()
            self.Wci = variable.Parameter(Wci_initializer)
            Wcf_initializer = initializers.Zero()
            self.Wcf = variable.Parameter(Wcf_initializer)
            Wco_initializer = initializers.Zero()
            self.Wco = variable.Parameter(Wco_initializer)

    def reset_state(self, pc = None, ph = None):
        self.pc = pc
        self.ph = ph

    def initialize_params(self, shape):
        self.Wci.initialize((self.mid, shape[2], shape[3]))
        self.Wcf.initialize((self.mid, shape[2], shape[3]))
        self.Wco.initialize((self.mid, shape[2], shape[3]))

    def initialize_state(self, shape):
        self.pc = Variable(self.xp.zeros((shape[0], self.mid, shape[2], shape[3]), dtype = self.xp.float32))
        self.ph = Variable(self.xp.zeros((shape[0], self.mid, shape[2], shape[3]), dtype = self.xp.float32))
        
    def __call__(self, x):
        if self.Wci.data is None:
            self.initialize_params(x.data.shape)

        if self.pc is None:
            self.initialize_state(x.data.shape)

        ci = F.sigmoid(self.Wxi(x) + self.Whi(self.ph) + F.scale(self.pc, self.Wci, 1))
        cf = F.sigmoid(self.Wxf(x) + self.Whf(self.ph) + F.scale(self.pc, self.Wcf, 1))
        cc = cf * self.pc + ci * F.tanh(self.Wxc(x) + self.Whc(self.ph))
        co = F.sigmoid(self.Wxo(x) + self.Who(self.ph) + F.scale(cc, self.Wco, 1))
        ch = co * F.tanh(cc)

        self.pc = cc
        self.ph = ch
        
        return ch


class MovingMnistNetwork(Chain):
    def __init__(self, sz=[256, 128, 128], n=256, directory=None):
        super(MovingMnistNetwork, self).__init__(
            e1 = ConvLSTM(n, sz[0], 5),
            e2 = ConvLSTM(sz[0], sz[1], 5),
            e3 = ConvLSTM(sz[1], sz[2], 5),
            p1 = ConvLSTM(n, sz[0], 5),
            p2 = ConvLSTM(sz[0], sz[1], 5),
            p3 = ConvLSTM(sz[1], sz[2], 5),
            last = L.Convolution2D(sum(sz), n, 1)
        )

        self.n = n
        self.directory = directory

    def save_image(self, arr, filename):
        img = chainer.cuda.to_cpu(arr)
        img = img * 255
        img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(filename)
        
    def __call__(self, x, t):
        self.e1.reset_state()
        self.e2.reset_state()
        self.e3.reset_state()

        We = self.xp.array([[i == j for i in range(self.n)] for j in range(self.n)], dtype=self.xp.float32)
        for i in range(x.shape[1]):

            # save input images
            if self.directory is not None:
                for j in range(x.shape[0]):
                    filename = self.directory + "input" + str(j) + "-" + str(i) + ".png"
                    self.save_image(x[j, i, :, :].data, filename)
                
            xi = F.embed_id(x[:, i, :, :], We)
            xi = F.transpose(xi, (0, 3, 1, 2))
            
            h1 = self.e1(xi)
            h2 = self.e2(h1)
            self.e3(h2)

        self.p1.reset_state(self.e1.pc, self.e1.ph)
        self.p2.reset_state(self.e2.pc, self.e2.ph)
        self.p3.reset_state(self.e3.pc, self.e3.ph)

        loss = None
        
        for i in range(t.shape[1]):
            xs = x.shape
            
            h1 = self.p1(Variable(self.xp.zeros((xs[0], self.n, xs[2], xs[3]), dtype=self.xp.float32)))
            h2 = self.p2(h1)
            h3 = self.p3(h2)

            h = F.concat((h1, h2, h3))
            ans = self.last(h)

            # save output and teacher images
            if self.directory is not None:
                for j in range(t.shape[0]):
                    filename = self.directory + "truth" + str(j) + "-" + str(i) + ".png"
                    self.save_image(t[j, i, :, :].data, filename)
                    filename = self.directory + "output" + str(j) + "-" + str(i) + ".png"
                    self.save_image(self.xp.argmax(ans[j, :, :, :].data, 0).astype(np.int32), filename)

            cur_loss = F.softmax_cross_entropy(ans, t[:, i, :, :])
            loss = cur_loss if loss is None else loss + cur_loss
            
        reporter.report({'loss': loss}, self)
        
        return loss


