k=4
for ii in range(500):
    ii=ii+1
    y=k**0.5
    i=0.3*y
    c=y-i
    deltk=0.1*k
    dk=i-deltk
    print(" year "+str(ii)+" k "+str(k)+" y "+str(y)+" c "+str(c)+" i "+str(i)+" deltk "+str(deltk)+" dk "+str(dk))
    k=k+dk
