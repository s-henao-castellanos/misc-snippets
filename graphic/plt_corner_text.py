def corner_text(s,x=0.02,y=0.95,ax=None):
    if ax == None:
        ax = plt.gca()
    plt.text(x,y,s,transform=plt.gca().transAxes,va="center",fontsize=13)