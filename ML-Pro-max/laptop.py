import pandas as pd
import numpy as np
#import csv
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import Normalizer
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import  mean_squared_error
import customtkinter as ctk

data = pd.read_csv("laptops.csv")
usedColumns=["CPU","CPU_GEN ","GPU_Capacity "
        ,"Screen_Size","OHE_DDR3","OHE_DDR4","OHE_DDR5",
        "RAM_Capacity ","Hard 1","Hard 2","OHE_Plastic",
        "OHE_G","OHE_H","OHE_HQ","OHE_U","OHE_X","OHE_2K","OHE_4K","OHE_FHD","OHE_HD",
        "OHE_Aluminum","OHE_Good","OHE_Very good","OHE_Not bad ","OHE_Bad",
        "OHE_HDD","OHE_SSD","OHE_M.2"]

def cleanandOneHot(data):
    ###cleaning dataset
    data=data.drop(columns="Name")
    data["GPU_Capacity "]=data["GPU_Capacity "].str.split(" ").str[0].astype(float)
    data["RAM_Capacity "]=data["RAM_Capacity "].str.split(" ").str[0].astype(float)
    data["Hard 1"]=data["Hard 1"].str.split(" ").str[0].astype(float)
    data["Hard 2"]=data["Hard 2"].str.split(" ").str[0].astype(float)  
    data["CPU"]=data["CPU"].str.split(" ").str[1].astype(int) 
    ### one hot encoder
    ohecl=['RAM(DDR3,DDr4,DDR5) ','Hard_Type (HDD,SSD,M.2)','Body_Type','CPU_Type ','Body_Status','Screen_Resolution ']
    data=pd.get_dummies(data=data,prefix='OHE',prefix_sep='_',columns=ohecl,dtype='int8')

    # Todo: sort data
    # idea : drop unused cols here then x will be  x=data.loc[:,data.columns!='Price']
    return data
data=cleanandOneHot(data)
###set xtrian ,ytrain and xtest,ytest  
y=data["Price"]


# 28
x=data[usedColumns]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=50, shuffle =True)

###Linear Regression
global linearmodel
linearmodel=LinearRegression()
linearmodel.fit(X_train,y_train)
pre=linearmodel.predict(X_test)
print ("Linear Regression : ",mean_absolute_error(y_test,pre))

### neural_network
lm = MLPRegressor(random_state=30, max_iter=5000)
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
print ("neural_network : ",mean_absolute_error(y_test,y_pred))





ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")
root=ctk.CTk()
root.columnconfigure(0,weight=1)
root.rowconfigure(0,weight=1)
root.rowconfigure(1,weight=1)
root.rowconfigure(2,weight=2)

fram1=ctk.CTkFrame(root)
fram2=ctk.CTkFrame(root)
fram3=ctk.CTkFrame(root)
fram4=ctk.CTkFrame(root)

root.geometry("410x460")
root.title("Laptop Price Predictor")
cpu_option=ctk.StringVar(root)
cpuType_option=ctk.StringVar(root)
gpuType_option=ctk.StringVar(root)
ram_option=ctk.StringVar(root)
hardType_option=ctk.StringVar(root)
bodyType_option=ctk.StringVar(root)
bodystate_option=ctk.StringVar(root)
screenRe_option=ctk.StringVar(root)

cp=["i 3","i 5","i 7","i 9","Rayzen 3","Rayzen 5","Rayzen 7","Rayzen 9"]
cpu_t=["U","G","M","H","HQ","X"]
ram=["DDR3","DDR4","DDR5"]
resloution=["HD","FHD","2k","4k"]
hard_type=["HDD","SSD","M.2"]
body_t=["plastic","Aluminum"]
body_s=["Very good","Good","Not bad","Bad"]
gpu_t=["Internal","External"]

global bt

name_lable=ctk.CTkLabel(fram1,text=f"Laptop name")
name_entry=ctk.CTkEntry(fram1,100,20,5,1)
cpu_gen_labble=ctk.CTkLabel(fram1,text=f"Cpu Gen")
cpu_gen_entry=ctk.CTkEntry(fram1,100,20,5,1)
GPU_Capacity_lable=ctk.CTkLabel(fram1,text=f"GPU Size")
GPU_Capacity_entry=ctk.CTkEntry(fram1,100,20,5,1)
Screen_Size_lable=ctk.CTkLabel(fram2,text=f"Screen Size")
Screen_Size_entry=ctk.CTkEntry(fram2,100,20,5,1)
RAM_Capacity_lable=ctk.CTkLabel(fram2,text=f"RAM")
RAM_Capacity_entry=ctk.CTkEntry(fram2,100,20,5,1)
Hard1_lable=ctk.CTkLabel(fram2,text=f"Hard 1 size")
Hard1_entry=ctk.CTkEntry(fram2,100,20,5,1)
Hard2_lable=ctk.CTkLabel(fram2,text=f"Hard 2 size")
Hard2_entry=ctk.CTkEntry(fram2,100,20,5,1)

cpu_option.set("CPU")
cpu=ctk.CTkOptionMenu(fram3,130,30,values=cp,variable=cpu_option)
ram_option.set("RAM")
Ram=ctk.CTkOptionMenu(fram3,130,30,values=ram,variable=ram_option)
cpuType_option.set("CPU Type")
cpu_type=ctk.CTkOptionMenu(fram3,130,30,values=cpu_t,variable=cpuType_option)
screenRe_option.set("Resolution")
Screen_Resolution=ctk.CTkOptionMenu(fram3,130,30,values=resloution,variable=screenRe_option)
hardType_option.set("Storage Type")
Hard_type=ctk.CTkOptionMenu(fram3,130,30,values=hard_type,variable=hardType_option)
bodyType_option.set("body Type")
body_type=ctk.CTkOptionMenu(fram3,130,30,values=body_t,variable=bodyType_option)
bodystate_option.set("body State")
body_Stat=ctk.CTkOptionMenu(fram3,130,30,values=body_s,variable=bodystate_option)
gpuType_option.set("GPU Type")
Gpu_type=ctk.CTkOptionMenu(fram3,130,30,values=gpu_t,variable=gpuType_option)

cpu.grid(row=0,column=6,pady=8,padx=9)
cpu_type.grid(row=1,column=6,pady=9)
Gpu_type.grid(row=2,column=6,pady=9)
Ram.grid(row=3,column=6,pady=9)
Hard_type.grid(row=4,column=6,pady=9)
body_type.grid(row=5,column=6,pady=9)
body_Stat.grid(row=6,column=6,pady=9)
Screen_Resolution.grid(row=7,column=6,pady=9)

fram1.grid(row=0 ,column=0,padx=10,pady=10,sticky="nsew")
fram2.grid(row=1 ,column=0,padx=10,pady=10,sticky="nsew")
fram3.grid(row=0,rowspan=2 ,column=1,padx=10,pady=10,sticky="nsew")
fram4.grid(row=2,column=0,columnspan=2,sticky="nsew",padx=10,pady=10)

name_lable.grid(row=0,column=0,padx=10,pady=8)
name_entry.grid(row=0,column=1,pady=8)

GPU_Capacity_lable.grid(row=1,column=0,padx=10,pady=8)
GPU_Capacity_entry.grid(row=1,column=1,padx=10,pady=8)

cpu_gen_labble.grid(row=2,column=0,padx=10,pady=8)
cpu_gen_entry.grid(row=2,column=1,padx=10,pady=8)

Hard1_lable.grid(row=0,column=0,padx=10,pady=8)
Hard1_entry.grid(row=0,column=1,pady=8)
Hard2_lable.grid(row=1,column=0,padx=10,pady=8)
Hard2_entry.grid(row=1,column=1,pady=8)
RAM_Capacity_lable.grid(row=2,column=0,padx=10,pady=8)
RAM_Capacity_entry.grid(row=2,column=1,padx=10,pady=8)
Screen_Size_lable.grid(row=3,column=0,padx=10,pady=8)
Screen_Size_entry.grid(row=3,column=1,padx=10,pady=8)
t_lable=ctk.CTkLabel(fram4,text=f"\t\t\t")
pridection=ctk.CTkEntry(fram4,width=100,height=30,corner_radius=5,border_width=1,placeholder_text="Price",font=('Arial', 16, 'bold'))
t_lable.grid(row=0,column=2)
pridection.grid(row=0 ,column=3)

def PredictNewprice():
    ### get values 
    lapname=name_entry.get()
    ### must be int
    cpuGen=int(cpu_gen_entry.get())
    gpuCapacity=GPU_Capacity_entry.get()
    hard1=Hard1_entry.get()
    hard2=Hard2_entry.get()
    ##must be int
    screensSize=float(Screen_Size_entry.get())
    ramCapacity=RAM_Capacity_entry.get()
    cpuOption=cpu_option.get()
    cputOption=cpuType_option.get()
    gputOption=gpuType_option.get()
    hardtOption=hardType_option.get()
    screenrOption=screenRe_option.get()
    bodytOption=bodyType_option.get()
    bodysOption=bodystate_option.get()
    ramtOption=ram_option.get()



    df=pd.DataFrame({'Name':[lapname],"CPU":[cpuOption],"CPU_Type ":[cputOption],"CPU_GEN ":[cpuGen] ,"GPU ":["gtx"]
                        ,"GPU_Type(builtin_0,separate_1)":[gputOption],"GPU_Capacity " : [gpuCapacity],"Screen_Size":[screensSize]
                        ,"Screen_Resolution ":[screenrOption],"RAM(DDR3,DDr4,DDR5) ":[ramtOption],"RAM_Capacity ":[ramCapacity]
                        ,"Hard 1":[hard1],"Hard 2":[hard2],"Hard_Type (HDD,SSD,M.2)":[hardtOption],"Body_Type":[bodytOption]
                        ,"Body_Status":[bodysOption],"Price":[0]})
    df= cleanandOneHot(df)
    #  add data columns to df cols(columns that OHE add it by falsy value)
    df=pd.concat([df,pd.DataFrame(columns=data.columns)],axis=0)
    df.replace(np.nan,0,inplace=True)
    #  delete any columns not exists in "data"
    df.drop([x for x in df.columns if x not in data.columns], axis=1 , inplace=True)
    df= df[usedColumns]

    pridection.delete(0,ctk.END)
    newlaptopPrice = linearmodel.predict(df)
    pridection.insert(0,f' $ { np.round(newlaptopPrice,2)[0]}')

bt=ctk.CTkButton(fram4,text="Predict",width=100,height=30,command=PredictNewprice)
bt.grid(row=0,column=0,pady=4,padx=5)
root.mainloop()