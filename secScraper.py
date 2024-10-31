import requests, pandas as pd
import matplotlib.pyplot as plt



class FinancialDataFetcher:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FinancialDataFetcher, cls).__new__(cls)
        return cls._instance
    def getFinancialKpiData(self, companyConcept):
        parsingData = companyConcept.json()['units']['USD']
        for data in parsingData:
            print(data)

#sec website requires a user-agent header in the request to see who is accessing their server
headers = {'User-Agent': "email@address.com"}

companyTickers = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=headers
    )

#review response/keys, converts response to json and then return keys in JSON
#print(companyTickers.json().keys())

#print first key(apple) data structure that appears in JSON file/ 
#cik_str is the central index key, unique id assigned by SEC, ticker is symbol for company, title is the name of company
companyDataVal1 = companyTickers.json()['5']
#print(companyDataVal1)

#extracts cik from company1 and stores it(_ _ _ _ _ _)
directCik = companyDataVal1['cik_str']

#converts entire JSon dictionary of companies into DataFrame
companyData = pd.DataFrame.from_dict(companyTickers.json(), orient='index')

#formats cik to pad cik_str with leading zeros to meet SEC format of 10 digits
companyData['cik_str'] = companyData['cik_str'].astype(str).str.zfill(10)

#prints out all companies, ticker, and title
#print(companyData)

#selects first row  of companyData and extracts cik_str which is the cik unique id. this will be used to fetch further company specific data on SEC
cik = companyData.iloc[0].cik_str
#sends request to SEC to fetch company metada
filingMetadata = requests.get(f'https://data.sec.gov/submissions/CIK{cik}.json',
    headers=headers)

#reviews json structure
#print(filingMetadata.json()['filings']['recent'].keys()) 

#creating a dataframe for easy viewing of recent filings data dictionary
allForms = pd.DataFrame.from_dict(filingMetadata.json()['filings']['recent'])
allForms.columns #columns in allForms
#print(allForms.head(50))

#10 q metadata
allForms.iloc[11]

#company facts data
companyFacts = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
    headers=headers)

#exploring json structure for companyFacts
companyFacts.json().keys()
companyFacts.json()['facts']
companyFacts.json()['facts'].keys()
#first value row of json that shows end, value, fiscal year quarter type of form when the form was filled
#print(companyFacts.json()['facts']['dei']['EntityCommonStockSharesOutstanding']['units']['shares'][0])

#extracts data about number of shares from companyFacts
#print(companyFacts.json()['facts']['dei']['EntityCommonStockShareOutstanding'])

#financial statement line items w/key financial items. has fiscal end data, value, accession number for each filing w the SEC, filed
companyFacts.json()['facts']['us-gaap']['AccountsPayable']
companyFacts.json()['facts']['us-gaap']['Revenues']
companyFacts.json()['facts']['us-gaap']['Assets']

companyConcept = requests.get(
    f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/Assets.json',
    headers=headers)

#structure of companyConcept data which has company financial metrics
companyConcept.json().keys()
companyConcept.json()['units']
companyConcept.json()['units'].keys()
#prints fy and end and val
print(companyConcept.json()['units']['USD'])

#dataFrame creation from filing data
assetsData = pd.DataFrame.from_dict((companyConcept.json()['units']['USD']))
assetsData.columns
#prints form
assetsData.form

#filters out non 10Q forms so only quarterly reports, reset_index reindexes DataFrame from 0 which cleans up index after filtering
assets10Q = assetsData[assetsData.form == '10-Q'].reset_index(drop=True)

#print(assetsData)
#plots end dates for 10Q vs assets value
#assets10Q.plot(x='end',y='val')
#plt.show()

dataFetcher = FinancialDataFetcher()
#dataFetcher.getFinancialKpiData(companyConcept)

    