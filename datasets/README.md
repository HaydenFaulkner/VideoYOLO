<h1 align="center">Datasets</h1>
<p align="center">
The currently supported datasets are - <a href="http://host.robots.ox.ac.uk/pascal/VOC/">Pascal VOC</a>, <a href="http://cocodataset.org/">MS-COCO</a>, <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-DET</a> and <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-VID</a>
</p>

<p align="center">The datasets will be stored in the following directory structure</p>
<pre>
VidDet/
└── datasets/
    ├── ImageNetDET (170.8 GB)
    ├── ImageNetVID (409.9 GB)
    ├── MSCoco (?? GB)
    ├── PascalVOC (?? GB)
    └── # version controlled files
</pre>

<p align="center">The datasets can be downloaded with their associated <code>.sh</code> script</p>

``` bash
VidDet/datasets$ . get_voc_dataset.sh
VidDet/datasets$ . get_coco_dataset.sh
VidDet/datasets$ . get_imgnetdet_dataset.sh
VidDet/datasets$ . get_imgnetvid_dataset.sh
```

<p align="center">These will make new directories (as shown in the structure above) and download into. If you want to use <b>symbolically linked</b> directories you will need to make these prior to running the scripts

<p align="center">.......</p>
<p align="center">If using the <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-VID</a> dataset please also download the <a href="https://drive.google.com/open?id=1-bvtqx71KNfNSi7twXbgBDdeoCA_use7"><code>val_motion_ious.json</code></a> file from my <a href="https://drive.google.com/open?id=1-bvtqx71KNfNSi7twXbgBDdeoCA_use7">Google Drive</a> and place it in the <code>VidDet/datasets/ImageNetVID/ILSVRC/</code> directory. This file allows for motion based evaluation, it will be generated when needed by <a href="https://github.com/HaydenFaulkner/VidDet/blob/ba28d3bf082c9e74a769bd2f1d7df47626e46b23/datasets/imgnetvid.py#L740"><code>generate_motion_ious()</code></a> in <a href="imgnetvid.py"><code>imgnetvid.py</code></a> if non existent however this is relatively time consuming, so we suggest downloading.</p>


<h2 align="center"></h2>
<h2 align="center">A Combined Dataset</h2>
<p align="center">It's possible to combine all four datasets into one larger dataset with the utilisation of the <a href="https://github.com/HaydenFaulkner/VidDet/blob/ba28d3bf082c9e74a769bd2f1d7df47626e46b23/datasets/combined.py#L16"><code>CombinedDetection()</code></a> dataset specified in <a href="combined.py"><code>combined.py</code></a></p>

<p align="center">Following ideas from <a href="https://github.com/philipperemy/yolo-9000">YOLO-9k</a> with utilising the <a href="https://wordnet.princeton.edu/">WordNet</a> structure classes have been manually matched across datasets, furthermore a <b>hierarchical tree structure</b> has been generated for the classes. This is visualised below and is specified in <a href="trees"><code>trees/</code></a>, with the main tree (inclusive of <a href="http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php">ImageNet-DET</a>) specified in <a href="trees/filtered.tree"><code>trees/filtered.tree</code></a></p>

<p align="center"><img src="../img/filtered_tree_det.svg"></p>

<h2 align="center"></h2>
<h2 align="center">Stats</h2>

<p align="center">Attained by running <a href="stats.py"><code>stats.py</code></a></p>

#### Training
| Wordnet ID  | Wordnet Synset | `voc trainval 07+12` | `coco train 17` | `det train` | `vid train` | `ytbb train` |
|-------------|----------------|---------|---------|--------|--------|--------|
| `n02691156` | airplane.n.01             | 1285    | 5129    | 1787    | 88708   | 173534 (10744) |
| `n02834778` | bicycle.n.01              | 1208    | 7056    | 1879    | 40326   | 149934 (10920) |
| `n01503061` | bird.n.01                 | 1820    | 10537   | 38206   | 150882  | 169014 (11655) |
| `n02858304` | boat.n.01                 | 1397    | 10575   |         |         | 176592 (10848) |
| `n02876657` | bottle.n.01               | 2116    | 24070   |         |         |                |
| `n02924116` | bus.n.01                  | 909     | 6061    | 3030    | 32715   | 163810 (10962) |
| `n02958343` | car.n.01                  | 4008    | 43532   | 10811   | 138791  | 1315915 (119935) |
| `n02121808` | domestic_cat.n.01         | 1616    | 4766    | 3514    | 61903   | 197538 (11565) |
| `n03001627` | chair.n.01                | 4338    | 38071   | 9642    |         |                |
| `n01887787` | cow.n.02                  | 1058    | 8014    |         |         | 148752 (10833) |
| `n03201035` | dining-room_table.n.01    | 1057    | 15695   |         |         |                |
| `n02084071` | dog.n.01                  | 2079    | 5500    | 74517   | 142089  | 187493 (11619) |
| `n02374451` | horse.n.01                | 1156    | 6567    | 2255    | 60757   | 183091 (11003) |
| `n03790512` | motorcycle.n.01           | 1141    | 8654    | 2624    | 37339   | 174895 (10525) |
| `n00007846` | person.n.01               | 15576   | 257249  | 60254   |         | 1113637 (75169) |
| `n13083023` | houseplant.n.01           | 1724    | 8631    |         |         |  135490 (9230) |
| `n02411705` | sheep.n.01                | 1347    | 9223    | 1883    | 42342   |                |
| `n04256520` | sofa.n.01                 | 1211    | 5779    | 1629    |         |                |
| `n04468005` | train.n.01                | 984     | 4570    | 1364    | 114381  | 187992 (11743) |
| `n03211117` | display.n.06              | 1193    | 5803    | 2820    |         |                |
| `n04490091` | truck.n.01                |         | 9970    |         |         | 177786 (10941) |
| `n06874185` | traffic_light.n.01        |         | 12841   | 1405    |         |                |
| `n03346898` | fireplug.n.01             |         | 1865    |         |         |                |
| `n06794110` | street_sign.n.01          |         | 1983    |         |         |                |
| `n03891332` | parking_meter.n.01        |         | 1283    |         |         |                |
| `n02828884` | bench.n.01                |         | 9820    | 1548    |         |                |
| `n02503517` | elephant.n.01             |         | 5484    | 2129    | 85340   | 170947 (10974) |
| `n02131653` | bear.n.01                 |         | 1294    | 3089    | 53989   | 180704 (10868) |
| `n02391049` | zebra.n.01                |         | 5269    | 1286    | 76941   |   15815 (1246) |
| `n02439033` | giraffe.n.01              |         | 5128    |         |         |   34860 (1902) |
| `n02769748` | backpack.n.01             |         | 8713    | 1588    |         |                |
| `n03963198` | platter.n.01              |         | 11265   |         |         |   87957 (5923) |
| `n02774152` | bag.n.04                  |         | 12341   |         |         |                |
| `n03815615` | necktie.n.01              |         | 6445    |         |         |                |
| `n02774630` | baggage.n.01              |         | 6112    |         |         |                |
| `n03397947` | frisbee.n.01              |         | 2681    |         |         |                |
| `n04228054` | ski.n.01                  |         | 6615    | 1075    |         |                |
| `n04251791` | snowboard.n.01            |         | 2676    |         |         |                |
| `n02778669` | ball.n.01                 |         | 6299    |         |         |                |
| `n04284869` | sport_kite.n.01           |         | 8801    |         |         |                |
| `n02799175` | baseball_bat.n.01         |         | 3273    |         |         |                |
| `n02800213` | baseball_glove.n.01       |         | 3747    |         |         |                |
| `n04225987` | skateboard.n.01           |         | 5536    |         |         | 149530 (10811) |
| `n02731900` | aquaplane.n.01            |         | 6091    |         |         |                |
| `n04409806` | tennis_racket.n.01        |         | 4803    |         |         |                |
| `n04592099` | wineglass.n.01            |         | 7839    |         |         |                |
| `n03438257` | glass.n.02                |         | 20574   |         |         |                |
| `n03383948` | fork.n.01                 |         | 5474    |         |         |                |
| `n04380346` | table_knife.n.01          |         | 7759    |         |         | 204597 (11278) |
| `n04284002` | spoon.n.01                |         | 6158    |         |         |                |
| `n02881193` | bowl.n.01                 |         | 14323   |         |         |                |
| `n07753592` | banana.n.02               |         | 9195    | 2239    |         |                |
| `n07739125` | apple.n.01                |         | 5776    | 2346    |         |                |
| `n07695965` | sandwich.n.01             |         | 4356    |         |         |                |
| `n07747607` | orange.n.01               |         | 6302    | 2241    |         |                |
| `n07714990` | broccoli.n.02             |         | 7261    |         |         |                |
| `n07730207` | carrot.n.03               |         | 7758    |         |         |                |
| `n07697537` | hotdog.n.02               |         | 2883    | 789     |         |                |
| `n07873807` | pizza.n.01                |         | 5807    | 949     |         |                |
| `n07639069` | doughnut.n.02             |         | 7005    |         |         |                |
| `n07613480` | trifle.n.01               |         | 6296    |         |         |                |
| `n02818832` | bed.n.01                  |         | 4192    |         |         |                |
| `n04446521` | toilet.n.02               |         | 4149    |         |         |  108129 (8991) |
| `n03642806` | laptop.n.01               |         | 4960    | 1646    |         |                |
| `n03793489` | mouse.n.04                |         | 2261    | 973     |         |                |
| `n04074963` | remote_control.n.01       |         | 5700    | 798     |         |                |
| `n03614007` | keyboard.n.01             |         | 2854    |         |         |                |
| `n04401088` | telephone.n.01            |         | 6420    |         |         |                |
| `n03761084` | microwave.n.02            |         | 1672    | 920     |         |                |
| `n03862676` | oven.n.01                 |         | 3334    |         |         |                |
| `n04442312` | toaster.n.02              |         | 225     | 679     |         |                |
| `n04553703` | washbasin.n.02            |         | 5609    |         |         |                |
| `n04070727` | refrigerator.n.01         |         | 2634    | 913     |         |                |
| `n02870092` | book.n.02                 |         | 24075   |         |         |                |
| `n03046257` | clock.n.01                |         | 6320    |         |         |                |
| `n04522168` | vase.n.01                 |         | 6577    |         |         |                |
| `n04148054` | scissors.n.01             |         | 1464    |         |         |                |
| `n04399382` | teddy.n.01                |         | 4729    |         |         |                |
| `n03483316` | hand_blower.n.01          |         | 198     | 733     |         |                |
| `n04453156` | toothbrush.n.01           |         | 1945    |         |         |                |
| `n02672831` | accordion.n.01            |         |         | 822     |         |                |
| `n02219486` | ant.n.01                  |         |         | 741     |         |                |
| `n02419796` | antelope.n.01             |         |         | 2598    | 59402   |                |
| `n02454379` | armadillo.n.01            |         |         | 684     |         |                |
| `n07718747` | artichoke.n.02            |         |         | 1470    |         |                |
| `n02764044` | ax.n.01                   |         |         | 1193    |         |                |
| `n02766320` | baby_bed.n.01             |         |         | 1760    |         |                |
| `n07693725` | bagel.n.01                |         |         | 1166    |         |                |
| `n02777292` | balance_beam.n.01         |         |         | 680     |         |                |
| `n02786058` | band_aid.n.01             |         |         | 821     |         |                |
| `n02787622` | banjo.n.01                |         |         | 778     |         |                |
| `n02799071` | baseball.n.02             |         |         | 598     |         |                |
| `n02802426` | basketball.n.02           |         |         | 950     |         |                |
| `n02807133` | bathing_cap.n.01          |         |         | 1694    |         |                |
| `n02815834` | beaker.n.01               |         |         | 1131    |         |                |
| `n02206856` | bee.n.01                  |         |         | 721     |         |                |
| `n07720875` | bell_pepper.n.02          |         |         | 1315    |         |                |
| `n02840245` | binder.n.03               |         |         | 628     |         |                |
| `n02870880` | bookcase.n.01             |         |         | 1242    |         |                |
| `n02879718` | bow.n.04                  |         |         | 826     |         |                |
| `n02883205` | bow_tie.n.01              |         |         | 820     |         |                |
| `n02880940` | bowl.n.03                 |         |         | 3488    |         |                |
| `n02892767` | brassiere.n.01            |         |         | 753     |         |                |
| `n07880968` | burrito.n.01              |         |         | 736     |         |                |
| `n02274259` | butterfly.n.01            |         |         | 4392    |         |                |
| `n02437136` | camel.n.01                |         |         | 2159    |         |                |
| `n02951585` | can_opener.n.01           |         |         | 569     |         |                |
| `n02970849` | cart.n.01                 |         |         | 2367    |         |                |
| `n02402425` | cattle.n.01               |         |         | 1366    | 55447   |                |
| `n02992211` | cello.n.01                |         |         | 976     |         |                |
| `n01784675` | centipede.n.01            |         |         | 786     |         |                |
| `n03000684` | chain_saw.n.01            |         |         | 623     |         |                |
| `n03017168` | chime.n.01                |         |         | 702     |         |                |
| `n03062245` | cocktail_shaker.n.01      |         |         | 502     |         |                |
| `n03063338` | coffee_maker.n.01         |         |         | 1439    |         |                |
| `n03085013` | computer_keyboard.n.01    |         |         | 1388    |         |                |
| `n03109150` | corkscrew.n.01            |         |         | 689     |         |                |
| `n03128519` | cream.n.03                |         |         | 1708    |         |                |
| `n03134739` | croquet_ball.n.01         |         |         | 1068    |         |                |
| `n03141823` | crutch.n.01               |         |         | 1168    |         |                |
| `n07718472` | cucumber.n.02             |         |         | 1064    |         |                |
| `n03797390` | mug.n.04                  |         |         | 3417    |         |                |
| `n03188531` | diaper.n.01               |         |         | 670     |         |                |
| `n03196217` | digital_clock.n.01        |         |         | 641     |         |                |
| `n03207941` | dishwasher.n.01           |         |         | 786     |         |                |
| `n02268443` | dragonfly.n.01            |         |         | 1569    |         |                |
| `n03249569` | drum.n.01                 |         |         | 2997    |         |                |
| `n03255030` | dumbbell.n.01             |         |         | 1524    |         |                |
| `n03271574` | electric_fan.n.01         |         |         | 922     |         |                |
| `n03314780` | face_powder.n.01          |         |         | 685     |         |                |
| `n07753113` | fig.n.04                  |         |         | 1073    |         |                |
| `n03337140` | file.n.03                 |         |         | 714     |         |                |
| `n03991062` | pot.n.04                  |         |         | 1827    |         |                |
| `n03372029` | flute.n.01                |         |         | 845     |         |                |
| `n02118333` | fox.n.01                  |         |         | 2681    | 38016   |                |
| `n03394916` | french_horn.n.01          |         |         | 610     |         |                |
| `n01639765` | frog.n.01                 |         |         | 2282    |         |                |
| `n03400231` | frying_pan.n.01           |         |         | 885     |         |                |
| `n02510455` | giant_panda.n.01          |         |         | 962     | 52984   |                |
| `n01443537` | goldfish.n.01             |         |         | 1899    |         |                |
| `n03445777` | golf_ball.n.01            |         |         | 656     |         |                |
| `n03445924` | golfcart.n.01             |         |         | 818     |         |                |
| `n07583066` | guacamole.n.01            |         |         | 798     |         |                |
| `n03467517` | guitar.n.01               |         |         | 2913    |         |                |
| `n03476991` | hair_spray.n.01           |         |         | 537     |         |                |
| `n07697100` | hamburger.n.01            |         |         | 705     |         |                |
| `n03481172` | hammer.n.02               |         |         | 671     |         |                |
| `n02342885` | hamster.n.01              |         |         | 876     | 40146   |                |
| `n03494278` | harmonica.n.01            |         |         | 657     |         |                |
| `n03495258` | harp.n.01                 |         |         | 1367    |         |                |
| `n03124170` | cowboy_hat.n.01           |         |         | 2146    |         |                |
| `n07714571` | head_cabbage.n.02         |         |         | 866     |         |                |
| `n03513137` | helmet.n.02               |         |         | 3784    |         |                |
| `n02398521` | hippopotamus.n.01         |         |         | 1107    |         |                |
| `n03535780` | horizontal_bar.n.01       |         |         | 572     |         |                |
| `n03584254` | ipod.n.01                 |         |         | 847     |         |                |
| `n01990800` | isopod.n.01               |         |         | 548     |         |                |
| `n01910747` | jellyfish.n.02            |         |         | 1662    |         |                |
| `n01882714` | koala.n.01                |         |         | 1070    |         |                |
| `n03633091` | ladle.n.01                |         |         | 658     |         |                |
| `n02165456` | ladybug.n.01              |         |         | 1126    |         |                |
| `n03636649` | lamp.n.02                 |         |         | 3304    |         |                |
| `n07749582` | lemon.n.01                |         |         | 1510    |         |                |
| `n02129165` | lion.n.01                 |         |         | 1000    | 32385   |                |
| `n03676483` | lipstick.n.01             |         |         | 682     |         |                |
| `n01674464` | lizard.n.01               |         |         | 5912    | 32302   |                |
| `n01982650` | lobster.n.02              |         |         | 2228    |         |                |
| `n03710721` | maillot.n.01              |         |         | 656     |         |                |
| `n03720891` | maraca.n.01               |         |         | 891     |         |                |
| `n03759954` | microphone.n.01           |         |         | 3477    |         |                |
| `n03764736` | milk_can.n.01             |         |         | 808     |         |                |
| `n03770439` | miniskirt.n.01            |         |         | 1152    |         |                |
| `n02484322` | monkey.n.01               |         |         | 8771    | 79053   |                |
| `n07734744` | mushroom.n.05             |         |         | 1425    |         |                |
| `n03804744` | nail.n.02                 |         |         | 1104    |         |                |
| `n03814639` | neck_brace.n.01           |         |         | 713     |         |                |
| `n03838899` | oboe.n.01                 |         |         | 705     |         |                |
| `n02444819` | otter.n.02                |         |         | 1073    |         |                |
| `n03908618` | pencil_box.n.01           |         |         | 754     |         |                |
| `n03908714` | pencil_sharpener.n.01     |         |         | 653     |         |                |
| `n03916031` | perfume.n.02              |         |         | 767     |         |                |
| `n03928116` | piano.n.01                |         |         | 1700    |         |                |
| `n07753275` | pineapple.n.02            |         |         | 832     |         |                |
| `n03942813` | ping-pong_ball.n.01       |         |         | 664     |         |                |
| `n03950228` | pitcher.n.02              |         |         | 1271    |         |                |
| `n03958227` | plastic_bag.n.01          |         |         | 873     |         |                |
| `n03961711` | plate_rack.n.01           |         |         | 714     |         |                |
| `n07768694` | pomegranate.n.02          |         |         | 1676    |         |                |
| `n07615774` | ice_lolly.n.01            |         |         | 871     |         |                |
| `n02346627` | porcupine.n.01            |         |         | 1226    |         |                |
| `n03995372` | power_drill.n.01          |         |         | 731     |         |                |
| `n07695742` | pretzel.n.01              |         |         | 1292    |         |                |
| `n04004767` | printer.n.02              |         |         | 668     |         |                |
| `n04019541` | puck.n.02                 |         |         | 560     |         |                |
| `n04023962` | punching_bag.n.02         |         |         | 730     |         |                |
| `n04026417` | purse.n.03                |         |         | 1659    |         |                |
| `n02324045` | rabbit.n.01               |         |         | 2194    | 40036   |                |
| `n04039381` | racket.n.04               |         |         | 656     |         |                |
| `n01495701` | ray.n.07                  |         |         | 1940    |         |                |
| `n02509815` | lesser_panda.n.01         |         |         | 1029    | 47935   |                |
| `n04116512` | rubber_eraser.n.01        |         |         | 972     |         |                |
| `n04118538` | rugby_ball.n.01           |         |         | 838     |         |                |
| `n04118776` | rule.n.12                 |         |         | 732     |         |                |
| `n04131690` | saltshaker.n.01           |         |         | 1049    |         |                |
| `n04141076` | sax.n.02                  |         |         | 991     |         |                |
| `n01770393` | scorpion.n.03             |         |         | 845     |         |                |
| `n04154565` | screwdriver.n.01          |         |         | 830     |         |                |
| `n02076196` | seal.n.09                 |         |         | 1892    |         |                |
| `n02445715` | skunk.n.04                |         |         | 1016    |         |                |
| `n01944390` | snail.n.01                |         |         | 909     |         |                |
| `n01726692` | snake.n.01                |         |         | 8696    | 33233   |                |
| `n04252077` | snowmobile.n.01           |         |         | 802     |         |                |
| `n04252225` | snowplow.n.01             |         |         | 785     |         |                |
| `n04254120` | soap_dispenser.n.01       |         |         | 694     |         |                |
| `n04254680` | soccer_ball.n.01          |         |         | 930     |         |                |
| `n04270147` | spatula.n.01              |         |         | 779     |         |                |
| `n02355227` | squirrel.n.01             |         |         | 925     | 48851   |                |
| `n02317335` | starfish.n.01             |         |         | 1175    |         |                |
| `n04317175` | stethoscope.n.01          |         |         | 924     |         |                |
| `n04330267` | stove.n.02                |         |         | 1859    |         |                |
| `n04332243` | strainer.n.01             |         |         | 845     |         |                |
| `n07745940` | strawberry.n.01           |         |         | 2247    |         |                |
| `n04336792` | stretcher.n.03            |         |         | 537     |         |                |
| `n04356056` | sunglasses.n.01           |         |         | 2671    |         |                |
| `n04371430` | swimming_trunks.n.01      |         |         | 955     |         |                |
| `n02395003` | swine.n.01                |         |         | 2413    |         |                |
| `n04376876` | syringe.n.01              |         |         | 989     |         |                |
| `n04379243` | table.n.02                |         |         | 8299    |         |                |
| `n04392985` | tape_player.n.01          |         |         | 1184    |         |                |
| `n04409515` | tennis_ball.n.01          |         |         | 759     |         |                |
| `n01776313` | tick.n.02                 |         |         | 673     |         |                |
| `n04591157` | windsor_tie.n.01          |         |         | 1367    |         |                |
| `n02129604` | tiger.n.02                |         |         | 1200    | 21712   |                |
| `n04487394` | trombone.n.01             |         |         | 739     |         |                |
| `n03110669` | cornet.n.01               |         |         | 973     |         |                |
| `n01662784` | turtle.n.02               |         |         | 2986    | 48935   |                |
| `n04509417` | unicycle.n.01             |         |         | 677     |         |                |
| `n04517823` | vacuum.n.04               |         |         | 622     |         |                |
| `n04536866` | violin.n.01               |         |         | 1254    |         |                |
| `n04540053` | volleyball.n.02           |         |         | 682     |         |                |
| `n04542943` | waffle_iron.n.01          |         |         | 714     |         |                |
| `n04554684` | washer.n.03               |         |         | 957     |         |                |
| `n04557648` | water_bottle.n.01         |         |         | 993     |         |                |
| `n04530566` | vessel.n.02               |         |         | 9250    | 59023   |                |
| `n02062744` | whale.n.02                |         |         | 1400    | 43662   |                |
| `n04591713` | wine_bottle.n.01          |         |         | 1391    |         |                |

#### Validation
| Wordnet ID  |   Wordnet Synset    | `voc test 07` | `coco val 17` | `det val` | `vid val` | `ytbb train` |
|-------------|---------------------|---------------|---------------|-----------|-----------|--------------|
| `n02691156` | airplane.n.01             | 311     | 143     | 101     | 45147   | 23172 (1404) |
| `n02834778` | bicycle.n.01              | 389     | 316     | 283     | 21602   | 18288 (1368) |
| `n01503061` | bird.n.01                 | 576     | 440     | 3310    | 100939  | 21468 (1488) |
| `n02858304` | boat.n.01                 | 393     | 430     |         |         | 22787 (1389) |
| `n02876657` | bottle.n.01               | 657     | 1025    |         |         |              |
| `n02924116` | bus.n.01                  | 254     | 285     | 218     | 16056   | 20532 (1410) |
| `n02958343` | car.n.01                  | 1541    | 1932    | 1212    | 85769   | 153294 (13371) |
| `n02121808` | domestic_cat.n.01         | 370     | 202     | 186     | 38991   | 24588 (1438) |
| `n03001627` | chair.n.01                | 1374    | 1791    | 1957    |         |              |
| `n01887787` | cow.n.02                  | 329     | 380     |         |         | 18612 (1397) |
| `n03201035` | dining-room_table.n.01    | 299     | 697     |         |         |              |
| `n02084071` | dog.n.01                  | 530     | 218     | 4288    | 63552   | 23547 (1458) |
| `n02374451` | horse.n.01                | 395     | 273     | 291     | 29036   |              |
| `n03790512` | motorcycle.n.01           | 369     | 371     | 267     | 9213    |              |
| `n00007846` | person.n.01               | 5227    | 11004   | 12823   |         |              |
| `n13083023` | houseplant.n.01           | 592     | 343     |         |         |              |
| `n02411705` | sheep.n.01                | 311     | 361     | 149     | 28231   |              |
| `n04256520` | sofa.n.01                 | 396     | 261     | 264     |         |              |
| `n04468005` | train.n.01                | 302     | 190     | 100     | 60619   |              |
| `n03211117` | display.n.06              | 361     | 288     | 421     |         |              |
| `n04490091` | truck.n.01                |         | 415     |         |         |              |
| `n06874185` | traffic_light.n.01        |         | 637     | 210     |         |              |
| `n03346898` | fireplug.n.01             |         | 101     |         |         |              |
| `n06794110` | street_sign.n.01          |         | 75      |         |         |              |
| `n03891332` | parking_meter.n.01        |         | 60      |         |         |              |
| `n02828884` | bench.n.01                |         | 413     | 227     |         |              |
| `n02503517` | elephant.n.01             |         | 255     | 121     | 27735   |              |
| `n02131653` | bear.n.01                 |         | 71      | 165     | 26107   |              |
| `n02391049` | zebra.n.01                |         | 268     | 72      | 8240    |              |
| `n02439033` | giraffe.n.01              |         | 232     |         |         |              |
| `n02769748` | backpack.n.01             |         | 371     | 419     |         |              |
| `n03963198` | platter.n.01              |         | 413     |         |         |              |
| `n02774152` | bag.n.04                  |         | 540     |         |         |              |
| `n03815615` | necktie.n.01              |         | 254     |         |         |              |
| `n02774630` | baggage.n.01              |         | 303     |         |         |              |
| `n03397947` | frisbee.n.01              |         | 115     |         |         |              |
| `n04228054` | ski.n.01                  |         | 240     | 171     |         |              |
| `n04251791` | snowboard.n.01            |         | 69      |         |         |              |
| `n02778669` | ball.n.01                 |         | 263     |         |         |              |
| `n04284869` | sport_kite.n.01           |         | 336     |         |         |              |
| `n02799175` | baseball_bat.n.01         |         | 146     |         |         |              |
| `n02800213` | baseball_glove.n.01       |         | 148     |         |         |              |
| `n04225987` | skateboard.n.01           |         | 179     |         |         |              |
| `n02731900` | aquaplane.n.01            |         | 269     |         |         |              |
| `n04409806` | tennis_racket.n.01        |         | 225     |         |         |              |
| `n04592099` | wineglass.n.01            |         | 343     |         |         |              |
| `n03438257` | glass.n.02                |         | 899     |         |         |              |
| `n03383948` | fork.n.01                 |         | 215     |         |         |              |
| `n04380346` | table_knife.n.01          |         | 326     |         |         |              |
| `n04284002` | spoon.n.01                |         | 253     |         |         |              |
| `n02881193` | bowl.n.01                 |         | 626     |         |         |              |
| `n07753592` | banana.n.02               |         | 379     | 237     |         |              |
| `n07739125` | apple.n.01                |         | 239     | 268     |         |              |
| `n07695965` | sandwich.n.01             |         | 177     |         |         |              |
| `n07747607` | orange.n.01               |         | 287     | 213     |         |              |
| `n07714990` | broccoli.n.02             |         | 316     |         |         |              |
| `n07730207` | carrot.n.03               |         | 371     |         |         |              |
| `n07697537` | hotdog.n.02               |         | 127     | 71      |         |              |
| `n07873807` | pizza.n.01                |         | 285     | 105     |         |              |
| `n07639069` | doughnut.n.02             |         | 338     |         |         |              |
| `n07613480` | trifle.n.01               |         | 316     |         |         |              |
| `n02818832` | bed.n.01                  |         | 163     |         |         |              |
| `n04446521` | toilet.n.02               |         | 179     |         |         |              |
| `n03642806` | laptop.n.01               |         | 231     | 166     |         |              |
| `n03793489` | mouse.n.04                |         | 106     | 175     |         |              |
| `n04074963` | remote_control.n.01       |         | 283     | 144     |         |              |
| `n03614007` | keyboard.n.01             |         | 153     |         |         |              |
| `n04401088` | telephone.n.01            |         | 262     |         |         |              |
| `n03761084` | microwave.n.02            |         | 55      | 78      |         |              |
| `n03862676` | oven.n.01                 |         | 143     |         |         |              |
| `n04442312` | toaster.n.02              |         | 9       | 46      |         |              |
| `n04553703` | washbasin.n.02            |         | 225     |         |         |              |
| `n04070727` | refrigerator.n.01         |         | 126     | 81      |         |              |
| `n02870092` | book.n.02                 |         | 1161    |         |         |              |
| `n03046257` | clock.n.01                |         | 267     |         |         |              |
| `n04522168` | vase.n.01                 |         | 277     |         |         |              |
| `n04148054` | scissors.n.01             |         | 36      |         |         |              |
| `n04399382` | teddy.n.01                |         | 191     |         |         |              |
| `n03483316` | hand_blower.n.01          |         | 11      | 52      |         |              |
| `n04453156` | toothbrush.n.01           |         | 105     |         |         |              |
| `n02672831` | accordion.n.01            |         |         | 56      |         |              |
| `n02219486` | ant.n.01                  |         |         | 52      |         |              |
| `n02419796` | antelope.n.01             |         |         | 285     | 11700   |              |
| `n02454379` | armadillo.n.01            |         |         | 50      |         |              |
| `n07718747` | artichoke.n.02            |         |         | 182     |         |              |
| `n02764044` | ax.n.01                   |         |         | 61      |         |              |
| `n02766320` | baby_bed.n.01             |         |         | 72      |         |              |
| `n07693725` | bagel.n.01                |         |         | 145     |         |              |
| `n02777292` | balance_beam.n.01         |         |         | 62      |         |              |
| `n02786058` | band_aid.n.01             |         |         | 74      |         |              |
| `n02787622` | banjo.n.01                |         |         | 54      |         |              |
| `n02799071` | baseball.n.02             |         |         | 77      |         |              |
| `n02802426` | basketball.n.02           |         |         | 54      |         |              |
| `n02807133` | bathing_cap.n.01          |         |         | 159     |         |              |
| `n02815834` | beaker.n.01               |         |         | 176     |         |              |
| `n02206856` | bee.n.01                  |         |         | 56      |         |              |
| `n07720875` | bell_pepper.n.02          |         |         | 144     |         |              |
| `n02840245` | binder.n.03               |         |         | 109     |         |              |
| `n02870880` | bookcase.n.01             |         |         | 170     |         |              |
| `n02879718` | bow.n.04                  |         |         | 55      |         |              |
| `n02883205` | bow_tie.n.01              |         |         | 118     |         |              |
| `n02880940` | bowl.n.03                 |         |         | 810     |         |              |
| `n02892767` | brassiere.n.01            |         |         | 161     |         |              |
| `n07880968` | burrito.n.01              |         |         | 74      |         |              |
| `n02274259` | butterfly.n.01            |         |         | 272     |         |              |
| `n02437136` | camel.n.01                |         |         | 137     |         |              |
| `n02951585` | can_opener.n.01           |         |         | 65      |         |              |
| `n02970849` | cart.n.01                 |         |         | 152     |         |              |
| `n02402425` | cattle.n.01               |         |         | 157     | 17951   |              |
| `n02992211` | cello.n.01                |         |         | 100     |         |              |
| `n01784675` | centipede.n.01            |         |         | 37      |         |              |
| `n03000684` | chain_saw.n.01            |         |         | 100     |         |              |
| `n03017168` | chime.n.01                |         |         | 122     |         |              |
| `n03062245` | cocktail_shaker.n.01      |         |         | 90      |         |              |
| `n03063338` | coffee_maker.n.01         |         |         | 44      |         |              |
| `n03085013` | computer_keyboard.n.01    |         |         | 283     |         |              |
| `n03109150` | corkscrew.n.01            |         |         | 54      |         |              |
| `n03128519` | cream.n.03                |         |         | 316     |         |              |
| `n03134739` | croquet_ball.n.01         |         |         | 150     |         |              |
| `n03141823` | crutch.n.01               |         |         | 103     |         |              |
| `n07718472` | cucumber.n.02             |         |         | 263     |         |              |
| `n03797390` | mug.n.04                  |         |         | 629     |         |              |
| `n03188531` | diaper.n.01               |         |         | 95      |         |              |
| `n03196217` | digital_clock.n.01        |         |         | 72      |         |              |
| `n03207941` | dishwasher.n.01           |         |         | 58      |         |              |
| `n02268443` | dragonfly.n.01            |         |         | 67      |         |              |
| `n03249569` | drum.n.01                 |         |         | 339     |         |              |
| `n03255030` | dumbbell.n.01             |         |         | 163     |         |              |
| `n03271574` | electric_fan.n.01         |         |         | 43      |         |              |
| `n03314780` | face_powder.n.01          |         |         | 92      |         |              |
| `n07753113` | fig.n.04                  |         |         | 232     |         |              |
| `n03337140` | file.n.03                 |         |         | 87      |         |              |
| `n03991062` | pot.n.04                  |         |         | 410     |         |              |
| `n03372029` | flute.n.01                |         |         | 150     |         |              |
| `n02118333` | fox.n.01                  |         |         | 181     | 11916   |              |
| `n03394916` | french_horn.n.01          |         |         | 85      |         |              |
| `n01639765` | frog.n.01                 |         |         | 126     |         |              |
| `n03400231` | frying_pan.n.01           |         |         | 113     |         |              |
| `n02510455` | giant_panda.n.01          |         |         | 46      | 8372    |              |
| `n01443537` | goldfish.n.01             |         |         | 185     |         |              |
| `n03445777` | golf_ball.n.01            |         |         | 117     |         |              |
| `n03445924` | golfcart.n.01             |         |         | 48      |         |              |
| `n07583066` | guacamole.n.01            |         |         | 69      |         |              |
| `n03467517` | guitar.n.01               |         |         | 190     |         |              |
| `n03476991` | hair_spray.n.01           |         |         | 96      |         |              |
| `n07697100` | hamburger.n.01            |         |         | 72      |         |              |
| `n03481172` | hammer.n.02               |         |         | 93      |         |              |
| `n02342885` | hamster.n.01              |         |         | 31      | 12472   |              |
| `n03494278` | harmonica.n.01            |         |         | 79      |         |              |
| `n03495258` | harp.n.01                 |         |         | 76      |         |              |
| `n03124170` | cowboy_hat.n.01           |         |         | 465     |         |              |
| `n07714571` | head_cabbage.n.02         |         |         | 167     |         |              |
| `n03513137` | helmet.n.02               |         |         | 843     |         |              |
| `n02398521` | hippopotamus.n.01         |         |         | 49      |         |              |
| `n03535780` | horizontal_bar.n.01       |         |         | 79      |         |              |
| `n03584254` | ipod.n.01                 |         |         | 93      |         |              |
| `n01990800` | isopod.n.01               |         |         | 61      |         |              |
| `n01910747` | jellyfish.n.02            |         |         | 144     |         |              |
| `n01882714` | koala.n.01                |         |         | 47      |         |              |
| `n03633091` | ladle.n.01                |         |         | 134     |         |              |
| `n02165456` | ladybug.n.01              |         |         | 104     |         |              |
| `n03636649` | lamp.n.02                 |         |         | 646     |         |              |
| `n07749582` | lemon.n.01                |         |         | 242     |         |              |
| `n02129165` | lion.n.01                 |         |         | 40      | 12809   |              |
| `n03676483` | lipstick.n.01             |         |         | 139     |         |              |
| `n01674464` | lizard.n.01               |         |         | 356     | 13419   |              |
| `n01982650` | lobster.n.02              |         |         | 111     |         |              |
| `n03710721` | maillot.n.01              |         |         | 99      |         |              |
| `n03720891` | maraca.n.01               |         |         | 101     |         |              |
| `n03759954` | microphone.n.01           |         |         | 348     |         |              |
| `n03764736` | milk_can.n.01             |         |         | 108     |         |              |
| `n03770439` | miniskirt.n.01            |         |         | 130     |         |              |
| `n02484322` | monkey.n.01               |         |         | 591     | 34972   |              |
| `n07734744` | mushroom.n.05             |         |         | 181     |         |              |
| `n03804744` | nail.n.02                 |         |         | 223     |         |              |
| `n03814639` | neck_brace.n.01           |         |         | 60      |         |              |
| `n03838899` | oboe.n.01                 |         |         | 76      |         |              |
| `n02444819` | otter.n.02                |         |         | 43      |         |              |
| `n03908618` | pencil_box.n.01           |         |         | 103     |         |              |
| `n03908714` | pencil_sharpener.n.01     |         |         | 94      |         |              |
| `n03916031` | perfume.n.02              |         |         | 86      |         |              |
| `n03928116` | piano.n.01                |         |         | 73      |         |              |
| `n07753275` | pineapple.n.02            |         |         | 136     |         |              |
| `n03942813` | ping-pong_ball.n.01       |         |         | 79      |         |              |
| `n03950228` | pitcher.n.02              |         |         | 126     |         |              |
| `n03958227` | plastic_bag.n.01          |         |         | 268     |         |              |
| `n03961711` | plate_rack.n.01           |         |         | 60      |         |              |
| `n07768694` | pomegranate.n.02          |         |         | 207     |         |              |
| `n07615774` | ice_lolly.n.01            |         |         | 71      |         |              |
| `n02346627` | porcupine.n.01            |         |         | 43      |         |              |
| `n03995372` | power_drill.n.01          |         |         | 59      |         |              |
| `n07695742` | pretzel.n.01              |         |         | 203     |         |              |
| `n04004767` | printer.n.02              |         |         | 55      |         |              |
| `n04019541` | puck.n.02                 |         |         | 66      |         |              |
| `n04023962` | punching_bag.n.02         |         |         | 80      |         |              |
| `n04026417` | purse.n.03                |         |         | 310     |         |              |
| `n02324045` | rabbit.n.01               |         |         | 117     | 12314   |              |
| `n04039381` | racket.n.04               |         |         | 90      |         |              |
| `n01495701` | ray.n.07                  |         |         | 111     |         |              |
| `n02509815` | lesser_panda.n.01         |         |         | 49      | 1149    |              |
| `n04116512` | rubber_eraser.n.01        |         |         | 262     |         |              |
| `n04118538` | rugby_ball.n.01           |         |         | 70      |         |              |
| `n04118776` | rule.n.12                 |         |         | 66      |         |              |
| `n04131690` | saltshaker.n.01           |         |         | 138     |         |              |
| `n04141076` | sax.n.02                  |         |         | 90      |         |              |
| `n01770393` | scorpion.n.03             |         |         | 48      |         |              |
| `n04154565` | screwdriver.n.01          |         |         | 119     |         |              |
| `n02076196` | seal.n.09                 |         |         | 153     |         |              |
| `n02445715` | skunk.n.04                |         |         | 70      |         |              |
| `n01944390` | snail.n.01                |         |         | 75      |         |              |
| `n01726692` | snake.n.01                |         |         | 517     | 13766   |              |
| `n04252077` | snowmobile.n.01           |         |         | 84      |         |              |
| `n04252225` | snowplow.n.01             |         |         | 60      |         |              |
| `n04254120` | soap_dispenser.n.01       |         |         | 102     |         |              |
| `n04254680` | soccer_ball.n.01          |         |         | 115     |         |              |
| `n04270147` | spatula.n.01              |         |         | 133     |         |              |
| `n02355227` | squirrel.n.01             |         |         | 40      | 24035   |              |
| `n02317335` | starfish.n.01             |         |         | 63      |         |              |
| `n04317175` | stethoscope.n.01          |         |         | 58      |         |              |
| `n04330267` | stove.n.02                |         |         | 191     |         |              |
| `n04332243` | strainer.n.01             |         |         | 76      |         |              |
| `n07745940` | strawberry.n.01           |         |         | 301     |         |              |
| `n04336792` | stretcher.n.03            |         |         | 74      |         |              |
| `n04356056` | sunglasses.n.01           |         |         | 610     |         |              |
| `n04371430` | swimming_trunks.n.01      |         |         | 210     |         |              |
| `n02395003` | swine.n.01                |         |         | 208     |         |              |
| `n04376876` | syringe.n.01              |         |         | 82      |         |              |
| `n04379243` | table.n.02                |         |         | 1682    |         |              |
| `n04392985` | tape_player.n.01          |         |         | 155     |         |              |
| `n04409515` | tennis_ball.n.01          |         |         | 174     |         |              |
| `n01776313` | tick.n.02                 |         |         | 49      |         |              |
| `n04591157` | windsor_tie.n.01          |         |         | 204     |         |              |
| `n02129604` | tiger.n.02                |         |         | 54      | 7767    |              |
| `n04487394` | trombone.n.01             |         |         | 125     |         |              |
| `n03110669` | cornet.n.01               |         |         | 157     |         |              |
| `n01662784` | turtle.n.02               |         |         | 229     | 21478   |              |
| `n04509417` | unicycle.n.01             |         |         | 91      |         |              |
| `n04517823` | vacuum.n.04               |         |         | 66      |         |              |
| `n04536866` | violin.n.01               |         |         | 97      |         |              |
| `n04540053` | volleyball.n.02           |         |         | 55      |         |              |
| `n04542943` | waffle_iron.n.01          |         |         | 75      |         |              |
| `n04554684` | washer.n.03               |         |         | 72      |         |              |
| `n04557648` | water_bottle.n.01         |         |         | 177     |         |              |
| `n04530566` | vessel.n.02               |         |         | 859     | 14148   |              |
| `n02062744` | whale.n.02                |         |         | 108     | 15928   |              |
| `n04591713` | wine_bottle.n.01          |         |         | 256     |         |              |
