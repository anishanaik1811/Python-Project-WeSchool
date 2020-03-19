#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
df.head()


# In[2]:


df=df.dropna(how='any')
sorted_data=df.sort_values(by= ["GDP Billions (USD) 2015","Gross Domestic Product Per Capita Income at Current Price (USD) 2015","Gross domestic product based on Purchasing-Power-Parity (PPP) valuation of Country GDP in Billions (Current International Dollar) 2015"],ascending = False)
sorted_data[:5]


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn import linear_model
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
df.head()
myData1 = pd.DataFrame(df, columns = ['Country Name', 'Population (Millions) 2010', 'Population (Millions) 2011', 'Population (Millions) 2012','Population (Millions) 2013','Population (Millions) 2014','Population (Millions) 2015']) 
rslt_df1 = myData1[myData1['Country Name'] == 'India']
rslt_df3=rslt_df1.values.tolist()
rslt_df4=rslt_df3[0]
rslt_df5 = rslt_df4[1:]
year=[2010,2011,2012,2013,2014,2015]
myData2 = pd.DataFrame(list(zip(year,rslt_df5)), 
               columns =['Year', 'Population']) 
myData2
reg=linear_model.LinearRegression()
reg.fit(myData2[['Year']],myData2.Population)
reg.predict([[2016]])


# In[ ]:


#the model predicted value 1315.25066667 and the actual value as per google was 1324.2 Million in 2016


# In[4]:


from matplotlib import pyplot as plt
from matplotlib import style
year=[2010,2010,2011,2012,2013,2014,2015]
myData1 = pd.DataFrame(df, columns = ['Country Name', 'Population (Millions) 2010', 'Population (Millions) 2011', 'Population (Millions) 2012','Population (Millions) 2013','Population (Millions) 2014','Population (Millions) 2015']) 
rslt_df1 = myData1[myData1['Country Name'] == 'India']
rslt_df3=rslt_df1.values.tolist()

myData2 = pd.DataFrame(df, columns = ['Country Name', 'GDP Billions (USD) 2010', 'GDP Billions (USD) 2011', 'GDP Billions (USD) 2012','GDP Billions (USD) 2013','GDP Billions (USD) 2014','GDP Billions (USD) 2015']) 
rslt_df2 = myData2[myData2['Country Name'] == 'India']
rslt_df4=rslt_df2.values.tolist()


style.use('ggplot')
plt.plot(year,rslt_df3[0],'g',label='Population')
plt.plot(year,rslt_df4[0],'c',label='GDP')


plt.title('Population Growth VS GDP 2010 to 2015')
plt.xlabel('2010 - 2015')
plt.ylabel('Population')
plt.legend
plt.grid(True)
plt.show()


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
df.head()
myData1 = pd.DataFrame(df, columns =['Country Name', 'GDP Billions (USD) 2010', 'GDP Billions (USD) 2011', 'GDP Billions (USD) 2012','GDP Billions (USD) 2013','GDP Billions (USD) 2014','GDP Billions (USD) 2015'])  
rslt_df1 = myData1[myData1['Country Name'] == 'India']
rslt_df3=rslt_df1.values.tolist()
rslt_df4=rslt_df3[0]
rslt_df5 = rslt_df4[1:]
year=[2010,2011,2012,2013,2014,2015]
myData2 = pd.DataFrame(list(zip(year,rslt_df5)), 
               columns =['Year', 'GDP']) 
myData2
reg=linear_model.LinearRegression()
reg.fit(myData2[['Year']],myData2.GDP)
reg.predict([[2016]])


# In[ ]:


#the model predicted value 2218.83533 and the actual value as per google was 2289.75 billions in 2016


# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
df.head()
myData1 = pd.DataFrame(df, columns =['Country Name', 'Gross Domestic Product Per Capita Income at Current Price (USD) 2010','Gross Domestic Product Per Capita Income at Current Price (USD) 2011','Gross Domestic Product Per Capita Income at Current Price (USD) 2012','Gross Domestic Product Per Capita Income at Current Price (USD) 2013','Gross Domestic Product Per Capita Income at Current Price (USD) 2014','Gross Domestic Product Per Capita Income at Current Price (USD) 2015'])  
rslt_df1 = myData1[myData1['Country Name'] == 'India']
rslt_df3=rslt_df1.values.tolist()
rslt_df4=rslt_df3[0]
rslt_df5 = rslt_df4[1:]
year=[2010,2011,2012,2013,2014,2015]
myData2 = pd.DataFrame(list(zip(year,rslt_df5)), 
               columns =['Year','PerCapita']) 
myData2
reg=linear_model.LinearRegression()
reg.fit(myData2[['Year']],myData2.PerCapita)
reg.predict([[2016]])


# In[ ]:


#the model predicted value 1692.956 and the actual value as per google was 1670 in 2016


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
pearsoncorr = df.corr(method='pearson')
pearsoncorr


# In[56]:





# In[59]:





# In[37]:


import matplotlib.pyplot as plt
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
bins= 'auto' 
myData1 = pd.DataFrame(df, columns =['Country Name', 'GDP Billions (USD) 2010', 'GDP Billions (USD) 2011', 'GDP Billions (USD) 2012','GDP Billions (USD) 2013','GDP Billions (USD) 2014','GDP Billions (USD) 2015'])  
rslt_df1 = myData1[myData1['Country Name'] == 'India']
rslt_df3=rslt_df1.values.tolist()
plt.hist(rslt_df3,bins,histtype='bar',rwidth=0.8,color='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[58]:


plt.bar([1,2,5,7],[5,2,7,8],label="abc",color='r')
plt.bar([2,4,6,8],[8,6,3,5],label="bcd",color='y')
plt.xlabel('bar number')
plt.ylabel('bar height')
plt.show()


# In[76]:


import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
sb.barplot(x=df['Country Name'].head(5),y=df['GDP Billions (USD) 2015'],data=df)


# In[80]:



import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
sb.barplot(x=df['Country Name'].head(11),y=df['GDP Billions (USD) 2015'],data=df)


# In[82]:



import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_csv (r'C:\Users\Anisha\Desktop\python project.csv')
sb.barplot(x=df['Country Name'].tail(11),y=df['GDP Billions (USD) 2015'],data=df)


# 

# In[ ]:




