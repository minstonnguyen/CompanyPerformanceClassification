from django.db import models

class Users(models.Model):
    username = models.CharField(max_length=50)
    firstName = models.CharField(max_length=50)
    lastName = models.CharField(max_length=50)
    email = models.EmailField(unique=True, default='default@example.com')
    #date joined, date last active, user admin add
    
class Companies(models.Model):
    cik = models.CharField(max_length=10)
    aliasName = models.CharField(max_length=50)
    name = models.CharField(max_length=100)
    ticker = models.CharField(max_length=6)
    endDate = models.CharField(max_length=10)
    startDate = models.CharField(max_length=10)
    fiscalYear = models.CharField(max_length=4)
    fiscalPeriod = models.CharField(max_length=2)
    accn = models.CharField(max_length=20) #accesion number
    #Finiancial KPI stats
    grossProfit = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    revenues = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    netIncomeLoss = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    assetsCurrent = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    liabilitiesCurrent = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    inventoryNet = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    costOfGoodsAndServicesSold = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    assets = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    totalLiabilities = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    liabilities = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    stockholdersEquity = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    weightedAverageNumberOfSharesOutstandingBasic = models.DecimalField(max_digits=20,decimal_places=2,null=True)
    currentSharePrice = models.DecimalField(max_digits=20,decimal_places=2,null=True)