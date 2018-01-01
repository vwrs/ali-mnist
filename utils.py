# -*- coding: utf-8 -*-
import sys

# show progress
# =====================
def progress(e,b,b_total,loss):
    sys.stdout.write("\r%3d: [%5d / %5d] loss: %f" % (e,b,b_total,loss))
    sys.stdout.flush()

def prog_ali(e,b,b_total,loss_g,loss_d,dx,dgz):
    sys.stdout.write("\r%3d: [%5d / %5d] G: %.4f D: %.4f D(x,Gz(x)): %.4f D(Gx(z),z): %.4f" % (e,b,b_total,loss_g,loss_d,dx,dgz))
    sys.stdout.flush()

