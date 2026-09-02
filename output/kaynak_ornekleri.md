# Kaynak Örnekleri — gözle kalite kontrolü

`probe_sources.py` çıktısından üretildi. Her config'ten 2 gerçek kayıt.
Önizlemeler 600 karakterde, cümle sonunda kesilmiştir.


---

## senti  ·  `turkish-nlp-suite/SentiTurca`

**YORUM — MusteriYorumlari + BuyukSinema'yı kapsıyor**


### senti/movies

Alanlar: `text:str`, `label:int`  
Metin alanı: **`text`**  ·  ortalama **455** karakter/kayıt

**Örnek 1** — 174 karakter

> İnandırıcılıktan oldukça uzak yapay bir fotoğraf sanatçılığı üzerine kurulmuş senaryosuyla, üçlü lezbiyen ilişkiyi kötü oyunculukla anlatamayan oldukça sıradan bir film. 2/10

**Örnek 2** — 470 karakter

> Recep İvedik ve Kutsal Damacana'yı Hollywood çekseydi nasıl olurdu sorusuna cevap veren bir film. Bir komedi filmi için orjinal bir konu, farklı bir tarzda da işlenmiş fakat filmdeki esprilerin tamamına yakını belaltı. Filmdeki bazı sahneler için gülmedim diyemem, izlerken de sıkmıyor fakat dediğim gibi seviye inanılmaz düşük ve bir süre sonra rahatsız ediyor her olayın belaltına bağlanması. Filmi beğenmek için izlerken sizin de seviyenizi baya bir düşürmeniz gerek.


### senti/e-commerce

Alanlar: `text:str`, `label:int`  
Metin alanı: **`text`**  ·  ortalama **96** karakter/kayıt

**Örnek 1** — 68 karakter

> Ürünleri 2025 olarak göndereceğiz dedikleri halde öyle gönderilmemiş

**Örnek 2** — 115 karakter

> Ürün görseldeki gibi. kalitelisini beğenmeedim. yumuşak ama çok ince diğer rengini de aldım o daha kaliteli ve kalı


### senti/hate

Alanlar: `baslik:str`, `text:str`, `label:int`  
Metin alanı: **`text`**  ·  ortalama **334** karakter/kayıt

**Örnek 1** — 81 karakter

> hayatta yapilan iyi kotu her$eyin gelecek ya$amlarini etkileyecegi inanci vardir.

**Örnek 2** — 143 karakter

> kocaman gözleri, kocaman ağızları ve minicik memeleri ile afacan erkek çocuğu gibi görünmeleri. saçları da küt ve daha kısası ise gachın derim.


---

## vitamin  ·  `turkish-nlp-suite/vitamins-supplements-reviews`

**YORUM — e-ticaret/takviye**


### vitamin/default

Alanlar: `product_name:str`, `brand:str`, `text:str`, `star:int`  
Metin alanı: **`text`**  ·  ortalama **61** karakter/kayıt

**Örnek 1** — 15 karakter

> güvenilir marka

**Örnek 2** — 49 karakter

> Hızlı kargo. Güzel paketlenmiş. Orijinal ürünler.


---

## forum  ·  `turkish-nlp-suite/ForumSohbetleri`

**FORUM/ARGO — asıl eksik olan register (11 alt-forum)**


### forum/donanimarsivi

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **1,513** karakter/kayıt

**Örnek 1** — liste: 5 mesaj, toplam 732 karakter, mesaj başına ~146

> Nasıl değiştirilir bilmiyorum

**Örnek 2** — liste: 9 mesaj, toplam 1,832 karakter, mesaj başına ~204

> Hangi firma? Netspeed yapmıştı bunu bize. Öylesine gitti 2 hafta gelmedi internet.


### forum/donanimhaber

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **1,824** karakter/kayıt

**Örnek 1** — liste: 2 mesaj, toplam 1,203 karakter, mesaj başına ~602

> AMD Athlon X4 860k İşlemci Gigabyte F2A68HM-DS2 Anakart Kingston HyperX XMP 8 GB 2x4 1600 Mhz CL9 Ram Sapphire R7 265 Dual-X 2 GB 256 Bit Ekran kartı WD Blue 1 TB 7200 RPM Harddisk Aerocool GT Black Edition 500W Kasa webdenaldan toplayabilirsin bütçeni 100 TL aşabilir.

**Örnek 2** — liste: 2 mesaj, toplam 327 karakter, mesaj başına ~164

> Maalesef bakkal işi geliştirme yapmışsınız. Oyun oynamıyor iseniz elinizdeki çağ dışı sistemi satıp yerine bu parçaları almanızı öneririm : H81M P33 G1840 4GB DDR3 1TB HDD + 120 GB SSD (yada tek 120'lik yeter derseniz sadece SSD alın.) Sistem çok hızlı olacaktır ve oyun oynamayacağınıza göre işinizi her türlü görecektir.


### forum/forumum

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **1,021** karakter/kayıt

**Örnek 1** — liste: 2 mesaj, toplam 641 karakter, mesaj başına ~320

> ReAL DRAW LA ReSiM KesMeYe YarDıM... LuTFeN yaRdIM Arkadaşlar Real Drawla ResiM Kesicem Resimin Etrafını Yapıyorum Ama Başladım Yerede GeliyoruM Yuvarlak Oluşuyor Ama O Alanı Kesmiyor Kesmem İcin Ne YapmalıyıM YarDım Eden Etmeyene Cok TşkR EdeRiM... RePLeRim SiziNL&

**Örnek 2** — liste: 1 mesaj, toplam 494 karakter, mesaj başına ~494

> as3 ile php veri gönderip yönlendirmek iyi günler arkadaşlar, flash as3 te ufak bir sorunum vardı as2 de flashtan php ye post yoluyla veri yolluyordum aynı zamanda o sayfaya yönleniyordu Kod: var veri:LoadVars = new LoadVars ;veri. kod = "gidecek veri";veri. send ; böyle basit bir şekilde butona tıkladığımda swfnin bulunduğu sayfada gonder. php açılıyordu hem de veriyi yollamış oluyordum as3 te bunu nasıl yapabilirim en basitiyle yardımcı olursanız sevinirim dün geceden beri bulduklarım...


### forum/iyinet

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **1,550** karakter/kayıt

**Örnek 1** — liste: 1 mesaj, toplam 1,474 karakter, mesaj başına ~1474

> @%1; Özel bir uygulama yazdırılacaktır. Ugulamanın özellikleri 1. Uygulamaya adres bilgileri veya anlık google map bilgileri ile kayıt olma 2. Üye kayıt olma 3. Eşleştirme ve haber verme 4. Üyeleri haritalar üzerinden gösterme Yukarıdaki maddelerin detay bilgileri olacak şekilde Android uygulama yazdırılacaktır. Şartlar 1. Sabit telefondan ulaşılabilen kişi veya kuruluşlar 2. Banka hesabı kendi adına olan 3. Fatura kesebilecek 4. Proje zaman planı verebilecek ve zaman planına sadık kalacak 5. Önden kesinlikle avans ödemesi yapılmayacaktır. 6. Kodlar Üniversite tarafından onaylandıktan sonra ilgili ödeme yapılacaktır.

**Örnek 2** — liste: 1 mesaj, toplam 1,678 karakter, mesaj başına ~1678

> Ülkemizde henüz birkaç aylık süredir yaygın olarak duyulmaya başlamış olsa da Spotify, yaklaşık 10 yıldan beri varlığını sürdürmekte olan müzik platformudur. Online olarak hizmetlerini sağlamakta olan platform, kolay kullanılabilir olan ara yüzü ile birlikte müzik dinlemek isteyen kişilere yasal yollardan bu hakkı tanıyan platformdur. Bilgisayar üzerinden kullanılabilir olmasının dışında cep telefonları üzerinden de kullanılabilen Spotify özel programlar sayesinde avantaj sağlamaktadır. Ancak cep telefonundaki şarkı listelerini Youtube üzerine geçirmek isteyen kişiler için sunulan programlar, bu ihtiyaçların karşılanmasını mümkün hale getirmektedir.


### forum/kadinlarklubu

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **8,896** karakter/kayıt

**Örnek 1** — liste: 2 mesaj, toplam 1,401 karakter, mesaj başına ~700

> tatil anlayisi kamp kurmak, yuruyus yapmak, ya da plajda sabahtan aksama yatmak olan bir cift olarak bu yil tatile gidecegimiz zamanda 10 aylik olacak oglumla nasil bir plan yapabiliriz dusunduk dusunduk bulamadik. 6 gunluk ve 12 gunluk iki farkli zamanda tatil planlamamiz gerekiyor, cocukla cok uzun saatler yolda gecirmek istemiyoruz, ama 2 yildir tatil yapamadim azicik yuzmeye ihtiyacim var, bir yandan tur seklinde cekiyor canimiz... Boyle kucuk bebeklerle yaptiginiz tatiller, olumlu ve olumsuz yonleri nelerdi? belleza

**Örnek 2** — liste: 5 mesaj, toplam 2,405 karakter, mesaj başına ~481

> terzi kendi söküğünü dikemezmiş ya aynen öyle bir turizmci olarak yüzlerce kişinin tatil organizasyonunu yapıyor ve kültür turları organize ediyorum kişisel tatillerimide kendim ayarlıyorum ama bebeğim olunca iş değişti çok tedirginim ve kafam karışık Ağustos ayında tatile çıkıcaz ve bebeğim o zaman 11 aylık olacak bebeğimle rahat edebileceğim onunda keyif alabileceği iyi bir otel istiyorum genelde hep belekte tatil yaparız, öncelikli tercih yerim belek sonra kemer veya side _DURU_24 BÜYÜK VE KÜÇÜK AŞKNA


### forum/memurlar

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **106,322** karakter/kayıt

**Örnek 1** — liste: 13 mesaj, toplam 1,666 karakter, mesaj başına ~128

> Çaykura bir önceki atamada yerleşen biri olarak yeni atanan arkadaşlara dilimiz döndüğünce bilgi vermek isteriz. herkese hayırlı olsun.

**Örnek 2** — liste: 24 mesaj, toplam 3,075 karakter, mesaj başına ~128

> umarım kesindir. candidman bunun üzerine bir soğuk su içer


### forum/tahribat

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **3,653** karakter/kayıt

**Örnek 1** — liste: 7 mesaj, toplam 1,590 karakter, mesaj başına ~227

> Android Cihazlarda Netv Gold uygulaması güvenilir mi. Yayınların Kalitesi çok güzel kullanmakta sakınca olabilir mi acaba https://netvgold.net.tr/app/ kontrol edebilecek birisi çıkarmı. Boş insan

**Örnek 2** — liste: 2 mesaj, toplam 763 karakter, mesaj başına ~382

> Ayarlardan geliştirici seçeneklerini aktif edip usb hata ayıklamayı açmayı unutma. https://github.com/Genymobile/scrcpy Herkesten, her şeyden umudunuzu kestiğiniz bir anda belki de kurtarıcı sizsinizdir.


### forum/technopatsosyal

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **1,765** karakter/kayıt

**Örnek 1** — liste: 33 mesaj, toplam 2,408 karakter, mesaj başına ~73

> Code Geass.

**Örnek 2** — liste: 9 mesaj, toplam 728 karakter, mesaj başına ~81

> Hocam fate serisini izleyebilirsiniz. Bazı sezonları seinen bazıları shounen. Birçok sezonu bulunuyor. Fate Zero ardından Fate/stay night: Unlimited Blade works izleseniz tatmin olursunuz gayet güzel bir seridir.


### forum/turkiyeforum

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **2,976** karakter/kayıt

**Örnek 1** — liste: 70 mesaj, toplam 9,069 karakter, mesaj başına ~130

> Evet şuan kullandığımız nicklerin anlamını, bizim için ne ifade ettiğini yada hikayesini buraya yazalım. İlginç hikayeleri bekliyorum. Buyrun, siz önden başlayın bende nickimin hikayesini anlatırım bi ara

**Örnek 2** — liste: 20 mesaj, toplam 1,577 karakter, mesaj başına ~79

> iyi ki geldin (: Sana mucizeler vaad etmedim; bir mucize verdim.. ' 01.12.2007


### forum/wardom

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **2,852** karakter/kayıt

**Örnek 1** — liste: 19 mesaj, toplam 3,926 karakter, mesaj başına ~207

> crack ı tekrar indir crack linki ayrıca verirsen tabiki indiririmmm

**Örnek 2** — liste: 7 mesaj, toplam 1,081 karakter, mesaj başına ~154

> Accounts yoksa parçalı olması daha iyi.. tek part büyük dosya free indirmede zaman aşımı yapıyor.. sonra ister istemez ağzımız bozuluyor bu şekil


### forum/wmaraci

Alanlar: `url:str`, `texts:list`  
Metin alanı: **`texts`**  ·  ortalama **781** karakter/kayıt

**Örnek 1** — liste: 3 mesaj, toplam 293 karakter, mesaj başına ~98

> while döngüsü ile yapabilirsin. katılıyorum

**Örnek 2** — liste: 1 mesaj, toplam 207 karakter, mesaj başına ~207

> as3 bilgisi iyi olan birinin yardımına ihtiyacım var. bir oyun içerisindeki bazı animasyonları ücretsiz veya küçük bir ücret karşılığında yapacak olan birileri varmı? Yapablecek olanlar pm ile ulaşabilirler.


---

## wiki  ·  `turkish-nlp-suite/temiz-Wiki`

**ANSİKLOPEDİK**


### wiki/default

Alanlar: `url:str`, `title:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **21,297** karakter/kayıt

**Örnek 1** — 100,459 karakter

> Cengiz Han (doğum adıyla Temuçin, 18 Ağustos 1227), Moğol İmparatorluğu'nun kurucusu ve ilk Kağanı olan Moğol komutan ve hükümdardır. Hükümdarlığı döneminde gerçekleştirdiği hiçbir savaşı kaybetmeyen Cengiz Han, dünya tarihinin en büyük askeri liderlerinden birisi olarak kabul edilmektedir. 13. yüzyılın başında Orta Asya'daki tüm göçebe bozkır kavimlerini birleştirip bir ulus haline getirerek Moğol siyasi kimliği çatısı altında toplamıştır. Cengiz Han, hükümdarlığı döneminde, 1206-1227 arasında, Kuzey Çin'deki Batı Xia ve Jin Hanedanı; Türkistan'daki Kara Hıtay, Maveraünnehir; Harezm, Horasan ...

**Örnek 2** — 518 karakter

> Film şu anlamlara gelebilir: Camlara yapıştırılarak içerinin görünmesini engelleyen bir tür ince yaprak Sinemacılıkta, bir oyunun bütününü taşıyan şerit veya şeritlerin bütünü Film (fotoğrafçılık), fotoğrafçılıkta, radyografide ve sinemacılıkta resim çekmek için kullanılan; selülozdan, saydam, bükülebilir şerit Film (sinema), sinema makinesiyle gösterilen eser, izleti Film (film), Samuel Beckett'in yazdığı tek senaryodan çekilen 1965 ABD yapımı film Total Film, 1997'den beri İngiltere'de yayımlanan sinema dergisi


---

## havadis  ·  `turkish-nlp-suite/Havadis`

**HABER**


### havadis/default

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **1,870** karakter/kayıt

**Örnek 1** — 2,211 karakter

> 1 Karat Kaç Gramdır? Bir Karat Kaç Gram? Değerli madenleri ölçmek için kullanılan bir ölçü birimi olarak karat ifade edilmektedir. Özellikle elmas başta olmak üzere pek çok değerli madeni ölçmek amaçlı önemli bir yere sahiptir. Bu doğrultuda karat ölçüm birimi aynı zamanda gram üzerinden de dönüşüm şansı vermektedir. Bu dönüşüm üzerinden bakıldığında ise bir karat 0,2 grama denk gelir. Böylece değerli taş madenlerin kütleleri hesabı yapılmak suretiyle buna uygun şekilde fiyatlandırması çıkarılır. Özellikle de güncel piyasa konusunda önemli yere sahiptir. 1 Karat Kaç Gramdır? Karat kelimesi günlük yaşamda çok sık karşılaşılan bir sözcük olarak öne çıkar.

**Örnek 2** — 2,334 karakter

> 1 Yumurta Kaç Gram Protein İçerir? Bir Yumurtada Ne Kadar Protein Var? Protein bakımından zengin yiyecekleri ön plana çıktığında bu konuda yumurta ön sıralarda yer alır. Hem uygun maliyeti hem de kolayca tüketimi ile birlikte et ve tavuğa göre daha fazla tercih edilmektedir. Çünkü kırmızı et ile tavuk gibi önemli oranda protein irtibatı sağladığını söylemek mümkün. Bu doğrultuda 100 gram üzerinden ele alındığında yumurta 13 gram protein sağlar. Tabii bu durum ortalama üzerinden bir yumurta ile alındığında 6,5 gram ettiğini söylemek gerekir. 1 Yumurta Kaç Gram Protein İçerir? Yumurta en sevilen ve aynı zamanda en çok tüketilen sağlıklı besin kaynakları içerisinde yer alır.


---

## ozenli  ·  `turkish-nlp-suite/OzenliDerlem`

**ÖZENLİ YAZILI — Havadis config'i DIŞLANACAK (havadis ile birebir aynı)**


### ozenli/GeziNotlari

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **3,556** karakter/kayıt

**Örnek 1** — 956 karakter

> Kaynakların sürdürülebilir kullanımı, iklim kriziyle mücadele ve ormanların korunmasının giderek önem kazandığı günümüzde Atlas dergisi de yeni bir adım atıyor ve Eylül 2023 sayısından itibaren okurlarının karşısına yepyeni bir kağıtla çıkıyor. Derginin iç sayfalarında kullanılan yeni kağıdın en önemli özelliği tamamen geri dönüştürülmüş malzemeden elde edilmesi, ayrıca geliştirilmiş standartları ve gramajıyla derginin görsel kalitesini daha da yükseltmesi. Alman Leipa firması ürettiği kağıtlarda hammadde olarak yüzde 100 atık kağıt kullanıyor, firma bu yolla yılda bir buçuk milyon tondan fazla geri kazanılmış kağıdı tekrar değerlendiriyor.

**Örnek 2** — 15,433 karakter

> Otomobillerden, yüksek sesten, karmaşa ve koşuşturmacadan uzakta, mavi sular, altın kumsallar ve geceleri yıldızlarla çevrili koylarda tatil yapmak isteyenlere mükemmel bir kaçış rotası: Beş ada; Gökçeada, Bozcaada, Marmara Adası, Avşa ve Paşalimanı. Ege'nin açık sularında el değmemiş koyları, göz alabildiğince uzanan kalabalıktan uzak plajları, eski Rum köyleri, zeytinlikleri ve şelaleleri ile Gökçeada, ada tatili yapmak isteyenler için her açıdan zengin bir tercih. Çanakkale il merkezinden 32, Gelibolu Yarımadası'ndaki Kabatepe Limanı'ndan 14 deniz mili uzaklıkta. Türkiye'nin en büyük adasının, yüzölçümü 289.


### ozenli/Havadis  ⛔ DIŞLANACAK

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **1,870** karakter/kayıt

**Örnek 1** — 2,211 karakter

> 1 Karat Kaç Gramdır? Bir Karat Kaç Gram? Değerli madenleri ölçmek için kullanılan bir ölçü birimi olarak karat ifade edilmektedir. Özellikle elmas başta olmak üzere pek çok değerli madeni ölçmek amaçlı önemli bir yere sahiptir. Bu doğrultuda karat ölçüm birimi aynı zamanda gram üzerinden de dönüşüm şansı vermektedir. Bu dönüşüm üzerinden bakıldığında ise bir karat 0,2 grama denk gelir. Böylece değerli taş madenlerin kütleleri hesabı yapılmak suretiyle buna uygun şekilde fiyatlandırması çıkarılır. Özellikle de güncel piyasa konusunda önemli yere sahiptir. 1 Karat Kaç Gramdır? Karat kelimesi günlük yaşamda çok sık karşılaşılan bir sözcük olarak öne çıkar.

**Örnek 2** — 2,334 karakter

> 1 Yumurta Kaç Gram Protein İçerir? Bir Yumurtada Ne Kadar Protein Var? Protein bakımından zengin yiyecekleri ön plana çıktığında bu konuda yumurta ön sıralarda yer alır. Hem uygun maliyeti hem de kolayca tüketimi ile birlikte et ve tavuğa göre daha fazla tercih edilmektedir. Çünkü kırmızı et ile tavuk gibi önemli oranda protein irtibatı sağladığını söylemek mümkün. Bu doğrultuda 100 gram üzerinden ele alındığında yumurta 13 gram protein sağlar. Tabii bu durum ortalama üzerinden bir yumurta ile alındığında 6,5 gram ettiğini söylemek gerekir. 1 Yumurta Kaç Gram Protein İçerir? Yumurta en sevilen ve aynı zamanda en çok tüketilen sağlıklı besin kaynakları içerisinde yer alır.


### ozenli/KulturHaritasi

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **5,874** karakter/kayıt

**Örnek 1** — 16,150 karakter

> Binbir Tv yazarları Işınla Bizi Scotty ve Uzun Çorap, 2017'nin dizilerini değerlendirdi. UÇ: Hımmm. Güzel bir bakış açısı. 2017 yılını şöyle bir düşündüğümde "Aman süper" diyemediğimi fark ettim. Fakat tek tek dizilerin üstünden geçerken belki unuttuklarımı hatırlarım, fikrim değişir. Bana diziler bakımından çok güçlü bir yıl gibi gelmedi. Bu yıla damga vuran diziye gelince... Dizilere şöyle bir bakıyorum, bir sürü kısa ömürlü dizi olmuş. Hem de çok. Senenin ilk çeyreğinde başlayıp halen süren 5 dizi var. İstanbullu Gelin, Fazilet Hanım ve Kızları, Fi, Payitaht ve Yeni Gelin... 13 dizi bitmiş. Fi'yi de dışarıda tutabiliriz çünkü reyting rekabetinde değil.

**Örnek 2** — 7,806 karakter

> Onur intihar etti. Alper Berna'yı evden kovdu. Çiçek'ten Berna'nın kendisi ve Leyla hakkında Çiçek'i doldurduğunu öğrendi. Ayla Nezihe'nin Canan olduğunu öğrendi. Alper ve Hakverdi, Leyla'ya tatile çıkalım diye baskı yapan Haşmet'i bertaraf etmek için şef Gürkan'ın cesedini buldular ve polise ihbar ettiler. Leyla Simge'nin Canan'a giysi odası videosuyla şantaj yaptığını öğrendi ve Simge'nin bilgisayarından videoyu sildi. Simge hamile kalmak için Akrokent'in fotoğrafçısıyla birlikte oldu. Burak'a o akşam çektiği fotoları gönderdi. Berna Haşmet'e Leyla ve Alper'in sevgili olduğunu söyledi. Bana Sevmeyi Anlat çok sağlam bir hikayeyle ilerliyor.


### ozenli/MasalMasal

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **2,147** karakter/kayıt

**Örnek 1** — 5,674 karakter

> Dış gebelik, döllenen yumurtanın sağlıklı ve doğru bir şekilde rahim içerisinde endometriyum tabakasına tutunamaması sonucu meydana gelen durumdur. Döllenen yumurta rahim dışında herhangi bir yere tutunursa buna dış gebelik / ektopik gebelikdenilmektedir. Dış gebelik durumu anne adaylarının hayatını riske sokacağı için bir an önce sonlandırılması gereken bir sağlık sorunudur. Anne adayına yapılan kan tahlilleri ve bazı testler sonucunda belirlenen hamilelik durumu her zaman sağlıklı bir şekilde ilerleyip sürecek anlamına gelmemektedir. Döllenen yumurtanın rahim ağzı veya fallop tüpleri içerisine bazı durumlarda ise karın içi vb.

**Örnek 2** — 4,553 karakter

> Haftalarca ve aylarca süren bekleyişinizin ardından dünyanın en büyük mucizesini kucağınıza aldığınızda bu heyecana Anne sıfatını da ekliyorsunuz. Yeni doğan bebeğinizin getirmiş olduğu mutluluk yanında büyük sorumluluklar da getiriyor. Ama merak etmeyin. Bilmeniz gereken her şeyi sizinle paylaşacağız. Doğum sonrası dönem, bebeğin doğumuyla başlar ve annenin vücudu neredeyse hamilelik öncesi durumuna döndüğünde sona erer. Bu dönem çoğunlukla altı-sekiz hafta sürer. Doğum sonrası dönem, annenin yeni bir anne olmanın gerektirdiği tüm değişikliklerle nasıl başa çıkacağını öğrenirken hem duygusal hem de aynı zamanda fiziksel olarak birçok değişiklikten geçmesini içerir.


### ozenli/Perdearkasi-Yorumlar

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **7,668** karakter/kayıt

**Örnek 1** — 6,464 karakter

> Bir süredir televizyonun altın çağını yaşıyoruz. Cesur senaryolar, yüksek bütçeler ve star oyuncular ile çekilen diziler, izleme alışkanlığını tamamen değiştiren dijital platformlar aracılığıyla seyircilere ulaşıyor. Artık alışılmışın dışında bir deneyime dönüşmüş dizi sektörü gerek temaları gerekse ele aldığı konuları anlatma kalitesiyle izleyicilerin izleme alışkanlıklarını yeniden yapılandırılıyor. Bu çerçevede televizyon hiç olmadığı kadar güncel ve değerli. İşte 2017 Yılının En İyi 10 TV Dizisi: - Twin Peaks: Return Televizyon dizilerindeki süregelen anlatıyı kökten değiştiren İkiz Tepeler 1990 yılında yayına başladı.

**Örnek 2** — 19,727 karakter

> Cineritüel yazarları 2017 yılı içerisinde ülkemizde vizyon yüzü gören filmler içerisinden en iyi 20 filmi belirledi. Yazarların kişisel listelerinde ortak bulunan filmlere göre oluşan listenin sıralamasında en çok oyu alan sıralamada kendisine yer buldu. Blade Runner 2049 ve Dunkirk filmlerinin puanları aynı olduğundan listemizde iki tane 20 numaralı film var bu sene. İşte Cineritüel'in 2017 en iyi 20 film seçkisi! - Manchester By The Sea / Yaşamın Kıyısında Kenneth Lonergan Kenneth Lonergan'ın üçüncü filmi Manchester by the Sea, hem metinsel açıdan hem de film dili, üslubu ve biçimi açısından patetik sıfatı ile bir yaşama uğraşı hikayesi yaratma konusunda önemli bir başarı kaydeder.


### ozenli/PopulerBilim

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **4,968** karakter/kayıt

**Örnek 1** — 5,516 karakter

> Tam da bir ay kadar önce Turgut Özal'ın ölüm haberini gazetede okuduğun anı şimdi bile hatırlıyorum. Politikaya yeni yeni ısınmamıza rağmen önemli bir şeyler olduğunu sezmişiz sanırım. Cafe Yelken miydi oranın adı? Ilık ve güneşli, Karadeniz baharının bir günüydü. Babamızın arkadaşı Ziver'i hatırlar mısın? Sen ortaokuldayken İskenderun gezisinde tanışmıştın. Çocuk olmana rağmen kendisine Ziver diye hitap etmen konusunda ısrar etmişti. Önce çok bocalamış sonra da sendeki hissini çok sevmiştin. Henüz amca, abi, bey olmadığın için başlamamışsındır ama sen de kendine hep MustafaCan denmesini isteyeceksin.

**Örnek 2** — 1,797 karakter

> Bu yazının başlığını aslında 21 Günde Alışmak Safsatası olarak belirlemiştim. Sonra Google'da arama yaptım; bununla ilgili size biraz istatistik vereyim. 21 günde alışmak diye arattığımda 1 milyon 360 bin sonuç bulundu. Vay! Biraz daha incelikli arama yapmak için 21 günde alışmak diye tırnak içinde arattığımda sadece iki sonuç geldi. Bunun üzerine 21 günde alışkanlık diye arattım: Tırnaksız yaklaşık 6 milyon sonuç, tırnak içinde ise 1.180 sonuç bulundu. Bu noktada yazının başlığına dönüp 21 Günde Alışkanlık Safsatası olarak değiştirdim. Neyse ki Google, ilk sonuçların az aşağısında kullanıcılar şunları da sordu diye bir şey getiriyor.


### ozenli/Serzenisler

Alanlar: `url:str`, `text:str`, `title:str`  
Metin alanı: **`text`**  ·  ortalama **572** karakter/kayıt

**Örnek 1** — 464 karakter

> Vodafone yıllardır iletişim olmayan, cep telefonu çekmeyen Ankara'nın köyünden bir haber. Ankara'da Etimesgut Belediyesi sınırları içerisinde olan Fevziye köyünde Vodafone cep telefonları hiç çekmiyor. Sanki çekmiyor. Sanki Ankara'da değil de Hakkari'de bir köyde. Vodafone iletişim gücü çok güçlü reklamlarına inat ve köye gitmekten hele de bu çağda. Vodafone iletişim gücü çok güçlü reklamlarına inat. Yillardir kim çözecek diye hala bekliyoruz. Ayip değil mi .!

**Örnek 2** — 251 karakter

> Rezervasyonum onaylanıp mail geldikten sonra Balıkesir Bandırma bölgesindeki ofisten araç teslim alamadık. Kurumsal kimlik namına hiç bir şey yok, tamamen fiyasko. Tavsiyem bu firmaya güvenip kesinlikle plan yapmayın, mağdur olmanız çok büyük ihtimal.


### ozenli/SusluTrendler

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **2,344** karakter/kayıt

**Örnek 1** — 1,153 karakter

> Sevilen oyuncu Serenay Sarıkaya ile gerçekleştirdiğimiz moda çekiminde, zamansız şıklığı ve ışıltıyı bir araya getirdik. 2024'ü #MagnificentWonders kampanyasıyla mucizeler ve harikalar diyarında karşılayan Bulgari'nin çok özel mücevherlerini sezonun heykelimsi formlarıyla buluşturduk ve Serenay'ın varlığını senenin yıldızı siyah-beyaz dengesiyle ikonikleştirmek istedik. Aralık sayımızda ayrıca, TikTok'un makyaj devrimi, sekiz şık Fransız moda markasının başarı dolu hikayeleri, Teknoloji ve Gelecek Danışmanı Elif Çetin'den, sektörlerden kişisel görüşlere, sizi şaşırtacak detaylarla 2024 beklentilerini bulabilirsiniz.

**Örnek 2** — 1,905 karakter

> Aralık ayın boyunca her Cumartesi ve Pazar günü Emaar'da her yerden sürpriz bir şekilde karşınıza çıkacak ve eşi benzeri görülmemiş Dev Kurabiye Adamlar! Üstelik bu kurabiye görünümlü eğlenceli karakterlerimiz, kendileri gibi harika görüntü ve lezzette ikramlarını ücretsiz bir şekilde Emaar ziyaretçilerine dağıtacaklar ve ağızlar kurabiye adamların kurabiyeleri ile tatlanacak. Yılbaşı ruhunun uyandırdığı, yaratıcılığın ortaya çıkacağı renkli atölyeler. Birbirinden yaratıcı ve eğlenceli atölyeler de Emaar yeni yıl masalında. Emaar misafirleri bu keyifli atölyelerde ücretsiz bir şekilde hem yaratıcılıklarını ortaya çıkarmayı deneyimleyecekler, hem de atölyelerde yaptıkları çalışmalar kendilerinde Emaar'dan bir hatıra olarak kalacak.


### ozenli/TeknoYazilar

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **2,421** karakter/kayıt

**Örnek 1** — 3,462 karakter

> Teknoloji ne kadar gelişse ve ne kadar hayatımızı kolaylaştırsa da beraberinde tehlikeleri de getirmeye devam ediyor. Teknolojinin neden olduğu tehlikeler içerisinde en öne çıkanları ise dolandırıcılar. Sürekli olarak yeni bir dolandırıcılık yöntemiyle karşı karşıya kalıyoruz. Son dönemde en sık kullanılan yöntem ise sahte numaralardan arayarak, kullanıcı bilgilerini ele geçirmek. Bu yazımızda da benzer şekilde kullanıcıların bilgilerini elde etmek için kullanılan 0850 220 0000 numarasından söz edeceğiz. Kullanıcılardan gelen mesajlara göre değerlendirme yapacak olursak, 0850 220 0000 numarası kullanıcıların bilgileri almak adına aramalar gerçekleştiriyor.

**Örnek 2** — 5,878 karakter

> 0xc0000142 hatası, seçili herhangi bir programın ilk çalıştırma aşamasında yaşanan ve programa erişimin kısıtlanmasına neden olan bir hatadır. Windows kullanıcıları bu hata ile karşılaştıkları zaman, hangi programı açmak istiyorsa istesinler program bir uyarı verecek ve sonrasında başlatılmadan kapanacaktır. Şunu düşünebilirsiniz, bu hata hangi uygulamalar ve oyunlarda yaşanıyor. Oyun ve uygulama isimleri vermemiz mümkün değil çünkü herhangi bir programda ya da herhangi bir oyun açılırken hatayı alabilirsiniz. Bir süreden sonra ciddi derecede can sıkıcı hale gelmeye ve programlara erişim sorunu yaşamanıza neden olan hata için çözüm yok değil.


### ozenli/ViralMedya

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **4,236** karakter/kayıt

**Örnek 1** — 3,009 karakter

> 0 Faktöriyel Neden 1'e Eşittir? 0 faktöriyelin 1'e eşit olması... permütasyon ve kombinasyon ile açık ve anlaşılır şekilde ispat ve izah edilebilirdir. konuya dönersek... işaret fonksiyonu gibi özel durumlarda da 0! "tanım" gereği 1 değerini verebiliriz. taban değeri sıfırdan büyük olan sayıların kuvvetleri sıfıra yaklaştığında, sayının değeri 1 e yaklaşır. örneğin hesap makinesinde bir pozitif sayının sürkeli karekökününü alırsanız sayının kuvveti sıfıra sayı ise 1 e yaklaşır. n elemanlı bir kümenin n elemanlı 1 tane alt kümesi vardır o da kendisidir. şimdi. * = (n!/n! ! dir. *n! sadeleşir. * 1/ !

**Örnek 2** — 5,448 karakter

> 0 Rakamını İlk Kim Buldu? sıfır rakamı, insanlık tarihinde uzunca bir süre kullanılmamış. şimdiye baktığımızdaysa sıfır rakamı olmadan işlem yapılması olanaksız. peki sıfırı gerçekten harezmi mi buldu? ya da diğer bilinen olarak yunanlar mı? bunu bulmak için yapılan birçok araştırma olmuş ve sonucunda sıfırı ne harezmi'nin, ne de yunanlıların bulduğu ortaya çıkmış. peki nasıl ve kimler tarafından bulunmuş? asırlar önce insanlar gündelik hayatlarında hesaplamalar yapma ihtiyacı hissetti. önce hepimizin çokça yaptığı gibi parmaklar kullanıldı, bu yöntem büyük hesaplamalarda yeterli gelmedi ve insanlar bu değerler için şekiller üretmeye başladı.


### ozenli/YazarinKaleminden

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **4,322** karakter/kayıt

**Örnek 1** — 4,587 karakter

> Harun: Fas'ta Hasan'la arkadaş olup güzel günler geçiren ve daha sonra Hasan'ın kardeşi Meryem'i kaçıran Hasan'ın sevdiği dostudur. Meryem: Zervali adında yaşlı bir tüccar ile evlenmek üzereyken kardeşinin çabaları sonucu evlenmekten kurtulup bir süre hasta haneye kapatılan Hasan'ın üvey kız kardeşidir. Hiba: Fas Kralı tarafından Hasan'a hediye edilen köle kızdır. Hasan görür görmez aşık olur ve bu aşk Hasan Fas'tan ayrılıncaya dek devam eder. Nur: Kahire'de hasan şehri gezerken tanıştığı Çerkez bir kızdır. Ayrıca Yavuz Sultan Selim'in yeğeni Alaaddin'in dul eşidir. İkisi arasında çok kısa bir zamanda aşk başlıyor ve evleniyorlar.

**Örnek 2** — 2,716 karakter

> İtalyan Köle: Aklın, sağduyu ve bilimin temsilcisi olan kişidir. Osmanlı tarafından ele geçirilmiş tutsak edilmiştir. Hoca lakaplı bilim adamına çalışmalarında yardımcı olmuştur. Hoca: İtalyan köle kimliğine bürünerek Batı'ya kaçıp Batı'ya yerleşmiş kişidir. Osmanlı alimidir. Toplumu değiştirecek bilimsel projelerin peşinden koşar. İcat ettiği silahın çamura saplanıp başarısızlıkla sonuçlanacak bir savaştan sonra öldürülmesinden korkarak Batı'ya yerleşmiştir. Osmanlı İmparatorluğu tarafından esir alınan İtalyan bir köle ile yaşadıkları olaylar sonucunda Hoca lakaplı bir alimin yerlerinden edilmesi ve benzerlikleri romanın konusunu oluşturur.


---

## akademik  ·  `turkish-nlp-suite/AkademikDerlem`

**AKADEMİK**


### akademik/makaleler

Alanlar: `dergi_ismi:str`, `title:str`, `url:str`, `pdf_url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **20,747** karakter/kayıt

**Örnek 1** — 35,612 karakter

> le, bireyler toplumsal iletişimde etrafındakileri etkilemek ve izlenim yönetmek için roller edinmektedirler. Bu roller gereği performanslarını sergilemekte ve bu sergileme için maskeler kullanmaktadırlar. Bu maskeler birey için birer benliğe dönüşmektedir; Çünkü bireyler ayrı ortamlarda ayrı amaçlarla ayrı ayrı roller oynamak durumunda kalmaktadırlar. 2. GÜNDELİK YAŞAMDA BENLİK SUNUMU VE DİJİTAL OYUNLAR İnsanlık iletişim kurmak, kendini ifade etmek, diğerleri tarafından da anlaşılır olmak için çok farklı yollar denemiştir. Bütün çabaları daha çok bireylerarasında kalmıştır. Değişen teknoloji ve gelişim gösteren hayat koşulları insana farklı kapıları aralamıştır.

**Örnek 2** — 27,315 karakter

> Karbon salımı son yıllarda artış göstermekte ve küresel ısınmanın en önemli sebeplerinden biri olarak gösterilmektedir. Türkiye karbon salımının temel nedeni ise enerji sektöründeki kullanımlardır. Bu çalışmada girdi çıktı tablosu ile hedef programlama modeli kurulmuştur. Uygulamada 11. Kalkınma Planı çerçevesinde sektörel enerji kullanımları ve CO2 salımları araştırılmıştır. RAS yöntemi kullanılarak 2012 girdi çıktı tablosu 2017 yılına güncellemiştir. Hedef programlama modelinde 2021,2022 ve 2023 yılları için GSYH değerleri hedef olarak alınmıştır. Programlamada yer alan kısıtlar; işgücü, sabit sermaye kullanımı, arz ve talep tutarları ve katsayılarıdır.


### akademik/akademik-ozetler

Alanlar: `dergi_ismi:str`, `title:str`, `url:str`, `pdf_url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **1,365** karakter/kayıt

**Örnek 1** — 938 karakter

> Bu çalışmada, örgütsel güvenin, örgütsel bağlılık ve işten ayrılma niyeti ile yakın bir ilişkiye sahip olduğu varsayılarak, aralarındaki ilişkinin gücünün ve yönünün karşılaştırmalı olarak analiz edilerek sonuçların ortaya konulması amaçlanmıştır. Varsayımın test edilmesi amacıyla nicel bir yöntem olan meta analiz uygulanmış, etki büyüklüğü değeri elde edebilmek amacıyla, Pearson (r) ve örneklem (n) değerleri kullanılmıştır. Yapılan taramada 141 çalışmaya ulaşılmış, seçim kıstaslarına uygun olan 84 çalışma analize dahil edilmiştir. Örgütsel güven ile örgütsel bağlılık arasında ES=0.55 değerinde güçlü ve pozitif, örgütsel güven ile işten ayrılma niyeti arasında ES=-0.

**Örnek 2** — 1,173 karakter

> Kültür ve popüler kültür kavramlarının yakından ilişkili olduğu olgular arasında spor ve özellikle futbol yer almaktadır. Futbolun yaygınlığının oluşturduğu güç onu bir endüstriye dönüştürerek, izleyici, taraftar ve holigan gibi kavramlar da yaratmıştır. Taraftar kavramının ise bugün geldiği noktada sosyal medya ile ilişkisi dikkat çekmektedir. Sosyal medya günümüzde taraftarın buluştuğu en popüler sosyal ortam olma özelliğiyle taraftar kültürünü en iyi yansıtan yerlerden biri haline gelmiştir. Bu çalışma popüler kültür, futbol ve taraftar ile sosyal medya kavramlarının aralarındaki ilişkiyi incelemeyi amaçlamaktadır.


### akademik/medikal-makaleler

Alanlar: `dergi_ismi:str`, `title:str`, `url:str`, `pdf_url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **4,850** karakter/kayıt

**Örnek 1** — 4,166 karakter

> Metformin insülin duyarlılığını artırarak etki gösteren ve diabetes mellitus tedavisinde çok yaygın olarak kullanılan oral antidiyabetiklerdendir. Güvenilir bir oral antidiyabetik olmasına rağmen uygunsuz kullanımı halinde çok ciddi yan etkilere yol açabilmektedir. Bu yan etkilerin başında laktik asidoz ve böbrek yetmezliği gelmektedir. Metformine bağlı laktik asidoz tablosu, böbrek veya karaciğer fonksiyon bozukluğu ya da enfeksiyon gibi eşlik eden bir durum yoksa genellikle ilacın yüksek dozda alınmasına bağlıdır. İntihar amaçlı ilacın yüksek dozda alınmasıyla ortaya çıkan ciddi laktik asidoz ve böbrek yetmezliği tablosu ölümcül olabilmektedir.

**Örnek 2** — 4,574 karakter

> Tiroid bezi dördüncü embriyonal haftada, dilin arka kısmında konumlanan foramen çekumdan köken alarak boyun ön kısmındaki anatomik lokalizasyonuna doğru göç etmeye başlar. Yedinci haftada hedeflenen anatomik lokalizasyona ulaşır. Bu iki bölge arasındaki bağlantıyı tiroglossal kanal sağlar ve bu kanal göç tamamlandıktan sonra atrofiye uğrar. Tiroglossal kanalın kaudal ucu kaybolmadan kalırsa piramidal lob meydana gelir. Piramidal lob sıklığı, seçilen çalışma popülasyonu ve tespit yöntemine göre değişmekle birlikte oldukça sık görülen bir varyasyondur. Yapılan çeşitli çalışmalarda, sıklığı %12 ile %75 arasında değişen oranlarda tespit edilmiştir.


### akademik/medikal-ozetler

Alanlar: `dergi_ismi:str`, `title:str`, `url:str`, `pdf_url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **1,260** karakter/kayıt

**Örnek 1** — 1,811 karakter

> Metformin güvenilir bir oral antidiyabetik olmakla birlikte yüksek dozda alınmasıyla ortaya çıkabilen laktik asidoz ve böbrek yetmezliği tablosu ölümcül olabilir. Bu çalışmada intihar amacıyla metformin alan olguların klinik seyri tartışılmıştır. Eylül 2009-Mayıs 2017 tarihleri arasında kliniğimizde takip edilen, intihar amaçlı yüksek doz metformin alan olgular değerlendirildi. Demografik verileri, biyokimya sonuçları, klinik seyirleri ve tedavileri dosyalarından kaydedildi. Toplam 15 olgunun 10 tanesi kadın, 5 tanesi erkekti. Medyan yaş kadınlarda 34 yıl (18-68), erkeklerde 38 yıl (23-58) saptandı.

**Örnek 2** — 940 karakter

> Çalışmamızda ultrasonografi ile piramidal lob sıklığını ve piramidal lob boyutları ile tiroid fonksiyon testleri arasında bir ilişki olup olmadığını tespit etmeyi amaçladık. Gereç ve Yöntem: Ekim 2015 ile ekim 2016 tarihleri arasında tiroid ultrasonografi için başvurmuş, erişkin yaş grubunda toplam 644 olgu çalışmamıza dahil edildi. Bulgular: Olgularımızın %15.2sinde (n=98) piramidal lob mevcuttu. Piramidal lob uzun boyutu ortalama 14.97±5.9 mm, kısa boyutu ortalama 3.99±5.1 mm idi. Piramidal lobu olan hastalar cinsiyete göre değerlendirildiğinde, kadın ve erkek cinsiyet arasında yaş, piramidal lob boyutları ve tiroid fonksiyonları açısından fark yoktu (p>0.


### akademik/bilkent-writings

Alanlar: `title:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **4,229** karakter/kayıt

**Örnek 1** — 3,909 karakter

> Marilyn Monroe akıllarda hep güzel sarışın olarak kaldı. Marilyn MONREO' yu bir kez de izlemek yerine okumak, tanımak, belki de böylesine güzel bir kadınla tanışmak biraz da sohbet etmek istedim. Başardım da ama bu kez onu hırçınlğıyla değil, hırçınlığının ona verdiği ızdırapla sevdim. Şaşırdım kendime; bir kadını izleyerek değil okuyarak sevdim. Kendi kendine yazdığı notlarda zayıflıklarını farkederek sevdim. Aslında onunla yalnız kalmak istemiştim. Sayfalara başlamadan önce sanatsal bir kişiliğin bana elçi olacağını tahmin edememiştim. Zaman zaman araya gireceğini hakkında her şeyi bildiğimi sandığım bir kadını yeniden tanıyacağımı düşünmek planlarım arasında değildi.

**Örnek 2** — 3,721 karakter

> Ben ölüm... Herkes korkar benden. Herkes tarafından dünyanın en kötü olayı olarak algılanırım. Gidenin ardından acının tarifsiz kaldığı cenazeler yapılır, benim yüzümden. İnsanlar, artık içlerine sığmayan acılarını dışa vurabilmek için yürekleri parçalayan ağıtlar yakarlar. Boğazlar düğümlenir, yutkunmak güçleşir. Gidenin acısı aylarca, yıllarca kalır geride kalanlarda. Zaman geçmek bilmez, yaşamak manasız hale gelir. Giden geminin yokluğuna bir türlü inandıramaz kendilerini limanda kalanlar. Bütün bu yaşattıklarıma rağmen, kulağınıza çok garip gelse de ben gerçekte iyi biriyim. Aslında benim bu kadar kötü algılanmamın sebebi en büyük düşmanım olan "yaşam"dır.


---

## cosmos  ·  `ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0`

**GENEL WEB**


### cosmos/default

Alanlar: `url:str`, `text:str`  
Metin alanı: **`text`**  ·  ortalama **4,927** karakter/kayıt

**Örnek 1** — 4,988 karakter

> Melek Mosso ve Serkan Sağdıç anlaşmalı olarak boşandı Komedyen Özgür Turhan'ın ETİ ile anlaşmasının feshedilmesine neden olan ırkçı tweet Unutulmaz karakterleriyle hafızalara kazınan Arif Erkin Güzelbeyoğlu'nun cenaze programı belli oldu Beyazıt Öztürk geri dönüyor: Yeni sergi yolda Kalp krizi sonrası ilk gelişme! Fatih Ürek'le ilgili umut veren detay Burcu Biricik kızı Luna ile görüntülendi: Anne-kız benzerliği şaşırttı Güllü'nün kızı Tuğyan Ülkem hakkındaki iddialara son nokta konuldu Kendi şarkısını duyunca dayanamadı! Sefo bornozuyla balkona fırladı Şahika Ercümen'den Gazze'ye destek: 107 ...

**Örnek 2** — 110 karakter

> Masaüstü bildirimlerimize izin vererek en son haberleri, analizleri ve derinlemesine içerikleri hemen öğrenin.


---

## fineweb2  ·  `HuggingFaceFW/fineweb-2`

**GENEL WEB — language_score + minhash_cluster_size ile filtrelenecek**


### fineweb2/tur_Latn

Alanlar: `text:str`, `id:str`, `dump:str`, `url:str`, `date:str`, `file_path:str`, `language:str`, `language_score:float`, `language_script:str`, `minhash_cluster_size:int`, `top_langs:str`  
Metin alanı: **`text`**  ·  ortalama **3,500** karakter/kayıt

**Örnek 1** — 485 karakter

> Her türlü oyunu tek sitede arayan oyun severler; Hoşgeldiniz! Burada bol eğlenceli, hertürlü online oyunu bulabilirsin. Günlük yüklenen oyunlarımızı takip et, yeni oyunlarımızı oynayan ilk sen ol! Erkekler için yarış oyunları, spor oyunları ve/veya macera oyunları mı aradın? Tam yerindesin! Kızlar için yemek oyunları, hayvancıklar ve/veya makyaj oyunları mı aradın? Burda bulursun! Beceri oyunları, zeka oyunları ve/veya birleştirme oyunları sevenler de burada aradıklarını bulurlar!

**Örnek 2** — 796 karakter

> Epilepsi ve Ben Resim Yarışması … Epilepsi ve Ben resim yarışması … “Epilepsi ve Ben” resim yarışması, çocuk epilepsi hastalarına sosyal destek sunmak amacıyla Türk Epilepsi ile Savaş Derneği tarafından organize edilmekte, sanofi-aventis Grup’un sponsorluğunda gerçekleştirilmektedir. Yarışmanın amacı, epilepsi konusunda farkındalığı artırmak, epilepsisi olan çocukların kendilerini ifade etmesine olanak yaratmak ve epilepsiden etkilenen bireyleri ve aileleri birleştirmektir. Resimlerin değerlendirmesi ve ödül alacak adayların belirlenmesi Türk Epilepsi ile Savaş Derneği’nin koordinasyonunda olup, sanofi-aventis Grup’tan bağımsızdır.
