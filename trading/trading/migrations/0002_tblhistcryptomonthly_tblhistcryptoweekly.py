# Generated by Django 3.2.6 on 2021-10-20 05:45

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TblHistCryptoWeekly',
            fields=[
                ('seq', models.AutoField(primary_key=True, serialize=False)),
                ('date', models.DateTimeField()),
                ('open', models.FloatField()),
                ('close', models.FloatField()),
                ('high', models.FloatField()),
                ('low', models.FloatField()),
                ('volume', models.FloatField()),
                ('fk_crypto_code', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='trading.tbscrypto')),
            ],
            options={
                'db_table': 'TBL_Hist_Crypto_Weekly',
            },
        ),
        migrations.CreateModel(
            name='TblHistCryptoMonthly',
            fields=[
                ('seq', models.AutoField(primary_key=True, serialize=False)),
                ('date', models.DateTimeField()),
                ('open', models.FloatField()),
                ('close', models.FloatField()),
                ('high', models.FloatField()),
                ('low', models.FloatField()),
                ('volume', models.FloatField()),
                ('fk_crypto_code', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='trading.tbscrypto')),
            ],
            options={
                'db_table': 'TBL_Hist_Crypto_Monthly',
            },
        ),
    ]
