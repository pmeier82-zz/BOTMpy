# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (C) 2011 by Philipp Meier, Felix Franke and
# Berlin Institute of Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#_____________________________________________________________________________
#
# Affiliation:
#   Bernstein Center for Computational Neuroscience (BCCN) Berlin
#     and
#   Neural Information Processing Group
#   School for Electrical Engineering and Computer Science
#   Berlin Institute of Technology
#   FR 2-1, Franklinstrasse 28/29, 10587 Berlin, Germany
#   Tel: +49-30-314 26756
#_____________________________________________________________________________
#
# Acknowledgements:
#   This work was supported by Deutsche Forschungs Gemeinschaft (DFG) with
#   grant GRK 1589/1
#     and
#   Bundesministerium für Bildung und Forschung (BMBF) with grants 01GQ0743
#   and 01GQ0410.
#_____________________________________________________________________________
#


from common import GdfFile, WriFile
import os.path as osp

EXP_DICT = {

    '19022008' : {

        'A' : [
            '19022008_0000.atf',
            '19022008_0001.atf',
            '19022008_0002.atf',
            '19022008_0003.atf',
            '19022008_0004.atf',
            '19022008_0005.atf',
            '19022008_0006.atf',
            '19022008_0007.atf',
            '19022008_0008.atf',
            '19022008_0009.atf',
            '19022008_0010.atf',
            '19022008_0011.atf',
            '19022008_0012.atf',
            '19022008_0013.atf',
        ],

        'B' : [
            '19022008_0024.atf',
            '19022008_0025.atf',
            '19022008_0026.atf',
            '19022008_0027.atf',
            '19022008_0028.atf',
            '19022008_0029.atf',
            '19022008_0030.atf',
            '19022008_0031.atf',
            '19022008_0032.atf',
            '19022008_0033.atf',
            '19022008_0034.atf',
            '19022008_0035.atf',
            '19022008_0038.atf',
            '19022008_0039.atf',
            '19022008_0040.atf',
            '19022008_0041.atf',
            '19022008_0042.atf',
            '19022008_0043.atf',
            '19022008_0044.atf',
            '19022008_0045.atf',
            '19022008_0046.atf',
            '19022008_0047.atf',
            '19022008_0048.atf',
            '19022008_0049.atf',
        ],

        'C' : [
            '19022008_0050.atf',
            '19022008_0051.atf',
            '19022008_0052.atf',
            '19022008_0053.atf',
            '19022008_0054.atf',
            '19022008_0055.atf',
            '19022008_0056.atf',
            '19022008_0057.atf',
            '19022008_0058.atf',
            '19022008_0059.atf',
            '19022008_0060.atf',
        ],
    },

    '24022009' : {
        'A' :[
            '24022009_0000.atf',
            '24022009_0001.atf',
            '24022009_0002.atf',
            '24022009_0003.atf',
            '24022009_0004.atf',
            '24022009_0005.atf',
            '24022009_0006.atf',
            '24022009_0007.atf',
            '24022009_0008.atf',
            '24022009_0009.atf',
            '24022009_0010.atf',
            '24022009_0011.atf',
            '24022009_0012.atf',
        ],

        'B' : [
            '24022009_0013.atf',
            '24022009_0014.atf',
            '24022009_0015.atf',
            '24022009_0016.atf',
            '24022009_0017.atf',
            '24022009_0018.atf',
            '24022009_0019.atf',
            '24022009_0020.atf',
            '24022009_0021.atf',
            '24022009_0022.atf',
            '24022009_0023.atf',
            '24022009_0024.atf',
            '24022009_0025.atf',
            '24022009_0026.atf',
            '24022009_0027.atf',
            '24022009_0028.atf',
            '24022009_0029.atf',
            '24022009_0030.atf',
            '24022009_0031.atf',
            '24022009_0032.atf',
            '24022009_0033.atf',
            '24022009_0034.atf',
            '24022009_0035.atf',
            '24022009_0036.atf',
            '24022009_0037.atf',
            '24022009_0038.atf',
            '24022009_0039.atf',
            '24022009_0040.atf',
            '24022009_0041.atf',
        ],
    },

    'HA25022009' : {
        'A' : [
            'HA25022009_0000.atf',
            'HA25022009_0001.atf',
            'HA25022009_0002.atf',
            'HA25022009_0003.atf',
            'HA25022009_0004.atf',
            'HA25022009_0005.atf',
            'HA25022009_0006.atf',
            'HA25022009_0007.atf',
            'HA25022009_0008.atf',
            'HA25022009_0009.atf',
        ],

        'B' : [
            'HA25022009_0010.atf',
            'HA25022009_0011.atf',
            'HA25022009_0012.atf',
            'HA25022009_0013.atf',
            'HA25022009_0014.atf',
            'HA25022009_0015.atf',
            'HA25022009_0016.atf',
            'HA25022009_0017.atf',
            'HA25022009_0018.atf',
            'HA25022009_0019.atf',
        ],

        'C' : [
            'HA25022009_0020.atf',
            'HA25022009_0021.atf',
            'HA25022009_0022.atf',
            'HA25022009_0023.atf',
            'HA25022009_0024.atf',
            'HA25022009_0025.atf',
            'HA25022009_0026.atf',
            'HA25022009_0027.atf',
            'HA25022009_0028.atf',
            'HA25022009_0029.atf',
            'HA25022009_0030.atf',
            'HA25022009_0031.atf',
            'HA25022009_0032.atf',
            'HA25022009_0033.atf',
            'HA25022009_0034.atf',
            'HA25022009_0035.atf',
            'HA25022009_0036.atf',
            'HA25022009_0037.atf',
            'HA25022009_0038.atf',
            'HA25022009_0039.atf',
        ],

        'D' : [
            'HA25022009_0041.atf',
            'HA25022009_0042.atf',
            'HA25022009_0043.atf',
            'HA25022009_0044.atf',
            'HA25022009_0045.atf',
            'HA25022009_0046.atf',
            'HA25022009_0047.atf',
            'HA25022009_0048.atf',
            'HA25022009_0049.atf',
            'HA25022009_0050.atf',
            # 51 to 56 are data of a different recording type!
            'HA25022009_0057.atf',
            'HA25022009_0058.atf',
            'HA25022009_0059.atf',
            'HA25022009_0060.atf',
            'HA25022009_0061.atf',
            'HA25022009_0062.atf',
            'HA25022009_0063.atf',
        ],

        'E' : [
            'HA25022009_0065.atf',
            'HA25022009_0066.atf',
            'HA25022009_0067.atf',
            'HA25022009_0068.atf',
            'HA25022009_0069.atf',
            'HA25022009_0070.atf',
            'HA25022009_0071.atf',
            'HA25022009_0072.atf',
            'HA25022009_0073.atf',
            'HA25022009_0074.atf',
            'HA25022009_0075.atf',
            'HA25022009_0076.atf',
            'HA25022009_0077.atf',
        ],
    },
}

##---CONSTANTS

WRIPATH = '/home/phil/Data/Alle/wri-8-11'
TRIALEN = 10000 * 100


##---FUNCTIONS

def trial_gdfs_from_wri(exp, block):

    # read wri file
    wri_name = osp.join(WRIPATH, '%s-%s000.wri' % (exp, block))
    wri_data = WriFile(wri_name).get_data()

    for i in xrange(len(EXP_DICT[exp][block])):
        atfname = EXP_DICT[exp][block][i]
        gdf_data = {}
        for u in wri_data.keys():
            trial_idx = (wri_data[u] > i * TRIALEN) * (wri_data[u] < (i + 1) * TRIALEN)
            gdf_data[ord(u) - 64] = wri_data[u][trial_idx]
        GdfFile.write_gdf(osp.join(WRIPATH, ''.join([atfname[:-4], '.gdf'])), gdf_data)


##---MAIN

if __name__ == '__main__':

    for exp in EXP_DICT:
        print 'exp:', exp
        for block in EXP_DICT[exp]:
            print 'block:', block
            trial_gdfs_from_wri(exp, block)
