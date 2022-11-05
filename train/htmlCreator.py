import numpy as np
from subprocess import call
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ipdb

header="""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
div {
    margin: 2px;
border: 50px solid #200000;
background-color: lightblue;
}
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
table {
    border-collapse: collapse;
    width: 100%;
}

th, td {
    text-align: left;
    padding: 8px;
}

tr:nth-child(even){background-color: #f2f2f2}
tr:nth-child(odd){background-color: #b0b0b0}

th {
    background-color: #000050;
    color: white;
}
</style>
</head>
<body>
<div>
<h1 align="center">The training progres</h1>
"""
imgHead = '<img src="'
imgTail='" alt="curve" style="width:50%;">'

tableTail="</table>"
tail="""
</div>
</body>
</html>
"""

def getHtmlTable(sep_iu,args):
        out=' '
        for item in sep_iu:
                #out=out+'<tr><td><a href=./'+args['jobid']+'/'+str(item[0])+'>'+ str(item[0])+'</a></td>'
                out=out+'<tr><td><a href=/cgi-bin/slide_view/s4m.py?exp='+args['jobid']+'&it='+str(item[0])+'>'+ str(item[0])+'</a></td>'
                for i in range(1,len(item)):
                        out=out+'<td> '+str(round(item[i]*100, 2))+'%</td>'
                out=out+'</tr>'
        return out

def paramPrint(args):
        out='<table><tr><th> Parameters</th><th> Values</th></tr>'
        for key, value in args.items() :
                if key=='best_record':
                        continue
                out=out+'<tr><td>'+key+'</td>'+'<td>'+str(value)+'</td></tr>'
        return out+'</table>'
def logHtml(test,args,Thead):
        file1 = open(args['dataset']+'_'+args['jobid']+'.html','w')
        file1.write(header)
        file1.write(imgHead)
        file1.write(args['jobid']+'.png')
        file1.write(imgTail)
        file1.write('<h2 align="center">Experiment Parameters</h2>')
        file1.write(paramPrint(args))
        file1.write('<h2 align="center">Test Class Wise Accuracy</h2>')
        file1.write(Thead)
        file1.write(getHtmlTable(test,args))
        file1.write(tableTail)
        file1.write(tail)
        file1.close()
        call(["scp", args['dataset']+'_'+args['jobid']+".html", "jobinkv@10.2.16.142:/home/jobinkv/Documents/r1/ijdar/"])
        print ('To view the training progres, please visit')
        print ('http://10.2.16.142/r1/ijdar/'+args['dataset']+'_'+args['jobid']+'.html')

def ploteIt(train_los,test_loss, accuracy,loss_g,loss_d,loss_e,aux_l, lr, printFrequncy,fname):
    t = np.arange(1, len(train_los)+1, 1)*printFrequncy
    train_los=np.asarray(train_los)
    fig, ax1 = plt.subplots()
    """
{'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}
    """
    ax1.set_xlabel('Iteration')
    plt.grid(True)
    ax1.set_ylabel('loss', color='tab:red')
    ax1.plot(t, train_los, color='tab:red',label='Train loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    if loss_g!=[]:
        ax1.plot(t, loss_g, color='tab:green',label='Global loss')
    if loss_d!=[]:
        ax1.plot(t, loss_d, color='tab:orange',label='Discriminative loss')
    if loss_e!=[]:
        ax1.plot(t, loss_e, color='tab:olive',label='Encoding loss')
    if aux_l!=[]:
        ax1.plot(t, aux_l, color='tab:gray',label='Auxilary loss')
    if lr!=[]:
        ax1.plot(t, lr, color='tab:brown',label='Learning rate')
    ax1.plot(t, test_loss, color='tab:pink',label='validation loss')
    ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Mean IoU', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(t, accuracy, color='tab:blue',label='Mean Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend()
    plt.title('Loss Mean accuracy curve')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname)
    call(["scp", fname+".png", "jobinkv@10.2.16.142:/home/jobinkv/Documents/r1/ijdar/"])

