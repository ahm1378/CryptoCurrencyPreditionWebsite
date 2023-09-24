from django.db import models

# Create your models here.
from django.db import models


# Create your models here.


class TbsCrypto(models.Model):
    crypto_name = models.CharField(max_length=255)
    pk_crypto_code = models.AutoField(primary_key=True)

    class Meta:
        db_table = "TBS_Crypto"


class TblHistCryptoDaily(models.Model):
    seq = models.AutoField(primary_key=True)
    fk_crypto_code = models.ForeignKey(TbsCrypto, on_delete=models.CASCADE)
    date =models.DateTimeField()
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    volume = models.FloatField()

    class Meta:
        db_table = "TBL_Hist_Crypto"


class TblHistCryptoWeekly(models.Model):
    seq = models.AutoField(primary_key=True)
    fk_crypto_code = models.ForeignKey(TbsCrypto, on_delete=models.CASCADE)
    date =models.DateTimeField()
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    volume = models.FloatField()

    class Meta:
        db_table = "TBL_Hist_Crypto_Weekly"


class TblHistCryptoMonthly(models.Model):
    seq = models.AutoField(primary_key=True)
    fk_crypto_code = models.ForeignKey(TbsCrypto, on_delete=models.CASCADE)
    date =models.DateTimeField()
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    volume = models.FloatField()

    class Meta:
        db_table = "TBL_Hist_Crypto_Monthly"

