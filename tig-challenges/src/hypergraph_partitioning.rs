use anyhow::{anyhow, Result};
use rand::
{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng, RngCore
};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};
use std::collections::HashSet;
use mt19937::{MT19937};

#[cfg(feature = "cuda")]
use crate::CudaKernel;
#[cfg(feature = "cuda")]
use cudarc::driver::*;
#[cfg(feature = "cuda")]
use std::{collections::HashMap, sync::Arc};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty 
{
    pub num_vertices:                                   usize,
    pub num_nodes:                                      usize,
    pub num_edges:                                      usize,
    pub num_blocks:                                     usize,
}

impl crate::DifficultyTrait<4> for Difficulty 
{
    fn from_arr(arr: &[i32; 4])                     -> Self 
    {
        return Self 
        {
            num_vertices:                               arr[0] as usize,
            num_nodes:                                  arr[1] as usize,
            num_edges:                                  arr[2] as usize,
            num_blocks:                                 arr[3] as usize
        };
    }

    fn to_arr(&self)                                -> [i32; 4] 
    {
        return [ self.num_vertices as i32, self.num_nodes as i32, self.num_edges as i32, self.num_blocks as i32 ];
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Solution 
{
    pub items:                                          Vec<usize>,
}

impl crate::SolutionTrait for Solution 
{
}

impl TryFrom<Map<String, Value>> for Solution 
{
    type Error                                      = serde_json::Error;

    fn try_from(v: Map<String, Value>)              -> Result<Self, Self::Error> 
    {
        return from_value(Value::Object(v));
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Challenge 
{
    pub seed:                                           [u8; 32],
    pub difficulty:                                     Difficulty,
    pub vertices:                                       Vec<u64>,
    pub hyperedge_indices:                              Vec<u64>,
    pub hyperedges:                                     Vec<Vec<u64>>,
    pub node_weights:                                   Vec<f32>,
    pub edge_weights:                                   Vec<f32>,
}

// TIG dev bounty available for a GPU optimisation for instance generation!
#[cfg(feature = "cuda")]
pub const KERNEL:                                   Option<CudaKernel> = None;

fn get_init_values(
    difficulty:                         &Difficulty, 
    seed:                               [u8; 32]
)                                                   -> (Vec<u64>, Vec<u64>, Vec<Vec<u64>>)
{
    let mut rng                                         = SmallRng::from_seed(StdRng::from_seed(seed).gen());
    let mut mt                                          = MT19937::new_with_slice_seed(&[2147483648u32,  3842733802u32,  2794635104u32,  1239646102u32,  3980727285u32,        3813458989u32,  3507724753u32,   574597546u32,  1436924229u32,  3796887534u32,        4031561246u32,  1197948434u32,  4088035066u32,  3886861965u32,      620331u32,        3343939013u32,   588374929u32,  2562865059u32,   246665089u32,  2431795719u32,        3025025982u32,  4170216049u32,  2715771869u32,  1768993125u32,  2519594765u32,         589951496u32,  3848468720u32,  1958695242u32,  1809735246u32,  2933756540u32,         762503952u32,   566763229u32,  2874674840u32,  3488149257u32,   959901508u32,        1970448903u32,  3229931700u32,  1730566471u32,  3825181373u32,   427908372u32,        2861036775u32,  1757190685u32,  2310844782u32,  3538624670u32,  3993968863u32,        2433577260u32,  3766900516u32,   378273054u32,  2281925123u32,  1683593272u32,        3710084241u32,  1048425864u32,   371642198u32,   676907449u32,  3005769642u32,        3273265679u32,  3541012911u32,  1449322563u32,  3020411413u32,  2523711641u32,         491741791u32,  2677888572u32,    20837343u32,  1472421846u32,  2538008582u32,         380903765u32,  1235761498u32,  2643772897u32,   809099657u32,  2029573017u32,        1487186790u32,  4240650582u32,  1429932763u32,  2591716315u32,   664076999u32,        2070700057u32,  1655114487u32,  3150206693u32,   482281117u32,  3621757770u32,        3059639774u32,   438749511u32,   792525582u32,  1968177573u32,   203538129u32,        1140428056u32,  4232771912u32,  2104289145u32,  3899587662u32,  1976942836u32,        1588399679u32,  1217433445u32,  2748615492u32,  2539624446u32,   266926730u32,        1872752266u32,  2088821824u32,   147022882u32,  2163878280u32,  2811438822u32,         762123869u32,  2143223263u32,   154042732u32,  1294355379u32,  3385104548u32,        3838259814u32,  3225448338u32,  1496534342u32,  2510668917u32,  3827990539u32,        1001301260u32,  3041073366u32,  1582246735u32,  2549571478u32,   824074668u32,        1492219140u32,  3310741797u32,  3294601662u32,  2680621125u32,   812917601u32,        1645316017u32,  2838956822u32,  3742843397u32,  1737217918u32,  2659662999u32,        3751432763u32,  1341626901u32,  2417613324u32,  2692938434u32,  3916262028u32,        2835483376u32,  1673763254u32,   815375401u32,  2655023393u32,  2979269462u32,        2378773084u32,  2799264825u32,   573566862u32,   848800453u32,  2894238199u32,        1239913143u32,  3309352033u32,  1418057127u32,  2879724848u32,  4271286294u32,         342657805u32,  3043319383u32,   246224984u32,  3493418441u32,  1115263128u32,        1291829501u32,  3049548241u32,  3099201434u32,  3750315688u32,  2880392253u32,        2670294748u32,   792511660u32,   284202065u32,  4074425016u32,   519364065u32,        2714008068u32,  3933731478u32,  3136830090u32,   761007657u32,  2620428800u32,        1065925097u32,  3521731243u32,  3497761006u32,  1075897408u32,   900828944u32,        2947485053u32,  3909082762u32,  1742918450u32,   883563659u32,  1275301281u32,        1370668583u32,   558848335u32,  2733395597u32,   955308866u32,  2917360150u32,        2142941386u32,   785090346u32,  2803095953u32,  2935651311u32,    24558038u32,        1751576135u32,  3123374648u32,  1248532638u32,   963275898u32,  3238733994u32,        3716710762u32,  4211951384u32,   797016769u32,  1697268427u32,  1527685628u32,        2045687195u32,   329888793u32,  1518518755u32,   305860601u32,  2702123162u32,          39802815u32,  1338244706u32,  2554533649u32,  2557757185u32,  1303237629u32,        1460701745u32,   577832531u32,  3511036371u32,  2129147752u32,  1792059825u32,        2738642190u32,  4241376315u32,  4190564662u32,   370768120u32,  2736589855u32,        1090883344u32,  4031232615u32,  1037910268u32,  2842276859u32,   769599018u32,        2535332238u32,   340899056u32,   399001770u32,  1951428933u32,  3223023323u32,        2693567597u32,   294798048u32,  3577745461u32,  1929552434u32,   337274491u32,        4165749262u32,  1597265313u32,  1925412259u32,  1439208297u32,  1520870260u32,        3042229537u32,  3821536280u32,  2880208781u32,  1739706131u32,  2474091234u32,        2800963059u32,  2070866703u32,    39409899u32,   807092718u32,   570098571u32,        3220240913u32,  1072266859u32,  2801152483u32,  1786098133u32,  1998054710u32,        1474351346u32,   271541466u32,   536332263u32,  2129859410u32,  3781222018u32,        1935117634u32,  1549151254u32,  3110335106u32,  2937053949u32,  2291328743u32,         274930387u32,  2147508065u32,    64939804u32,  1857203495u32,  3976537573u32,        3936900873u32,  3898927670u32,  3877430889u32,  3746258459u32,  1052565405u32,        3355266271u32,  1885817586u32,   338811057u32,  4189888221u32,  3188108084u32,        1909009212u32,  3723884037u32,  2830326354u32,  2473174379u32,  1087131865u32,        4237656501u32,  1479921940u32,  3818205703u32,   976852940u32,   205852117u32,        1569913139u32,  1549639231u32,  2414463228u32,   287338720u32,  1310168822u32,        1168527316u32,   497547986u32,  1239690708u32,   904512392u32,  2414230180u32,        3889048411u32,  3730351160u32,  1741868403u32,  1885641910u32,   887168660u32,          36792545u32,  2133387017u32,  4269553118u32,  2911297043u32,  2751192571u32,        3500631238u32,  3244240109u32,  1521555780u32,   281687482u32,   489641166u32,        1657088583u32,   676065630u32,  2153418490u32,  4028007991u32,  1323128015u32,        1444528364u32,  2948748498u32,  3926842287u32,  3938061400u32,  2189574807u32,        3541332052u32,  1903402642u32,  1631176139u32,   293339401u32,  3408070291u32,        3038107252u32,  2154227620u32,  3568324901u32,   846666135u32,  2592881453u32,         629464278u32,   416431015u32,  3921681808u32,  1930221101u32,  2133946643u32,        3060669179u32,  1646443938u32,   193888495u32,  3690722249u32,   419357707u32,        1565500351u32,   343220884u32,  2407986820u32,  2425700992u32,  3022656522u32,        2426011064u32,   248253022u32,  1291432716u32,  1073775760u32,  2412963699u32,        3744413261u32,  3961021709u32,  3361172631u32,  3176311219u32,  1425377671u32,        3117428901u32,  3250849524u32,  2045461386u32,  2076280707u32,   843910360u32,        2837804101u32,   301047632u32,   802966542u32,  1560088712u32,  2845926513u32,        3098387596u32,  4098451179u32,  2093533533u32,  3652540597u32,  1706120656u32,        3216980749u32,  3253478425u32,  4174983337u32,  3918881908u32,  2398651770u32,        1972301467u32,  2680372835u32,  2920543282u32,  1112609603u32,  2646779191u32,           3583526u32,   393749209u32,  4278204130u32,  1767992180u32,  1534803259u32,        1461370107u32,  1222623021u32,  3294852630u32,  2186699494u32,   964337179u32,        4077444554u32,   546629377u32,  2469522897u32,   841594760u32,  3216962540u32,        3450865861u32,  3731679548u32,   485242317u32,  1914898177u32,  1025037197u32,        3531227386u32,  4014876719u32,  1386302867u32,  1650719894u32,   516923199u32,         547166844u32,  2658311792u32,  1753152619u32,  1593552347u32,  2624965972u32,        3775945346u32,  1130062510u32,  2249580933u32,  3683661617u32,  3992166801u32,         944751450u32,  2500977361u32,   894766516u32,  3692587114u32,  2652816747u32,         743089786u32,   424814387u32,  3350739194u32,   187042739u32,  1675989893u32,        2807289330u32,  1504710498u32,  2790148155u32,  3611121645u32,  2266326836u32,         831816758u32,   131854414u32,  3322311174u32,   641626719u32,  3025511399u32,         728378527u32,  3269319085u32,  3797414838u32,   160570128u32,  3316684965u32,        2545537817u32,  1538942363u32,  3532641949u32,  3637936675u32,  3899808098u32,        3836822800u32,  1958937633u32,  2452084274u32,  3978187870u32,  3320793614u32,        3414746490u32,  2375145956u32,  4018383923u32,  1315982609u32,  2741768878u32,        3328331767u32,  2824096180u32,  1869940436u32,  2602424399u32,  2000130114u32,        3803183359u32,   269063464u32,   600521705u32,  3311859187u32,   708263863u32,        2596532642u32,  1422616313u32,  3889375014u32,  3393744568u32,  1360558073u32,        1392775174u32,  2030087957u32,  3024982637u32,  2369894496u32,  1966062525u32,         400479275u32,   653950058u32,  2985814358u32,   154622549u32,  1685281126u32,         114998298u32,  1960486971u32,  3689831580u32,   336259455u32,   584007679u32,        2518633598u32,  2291322576u32,  3332080654u32,  1981683070u32,  1734756807u32,        3864472453u32,  2304940311u32,  2061072649u32,   752095477u32,  1916071574u32,        1655200573u32,   957447697u32,  1231532294u32,  1230039010u32,  3562438051u32,         608123471u32,  1228846303u32,  2028580203u32,  3924895460u32,  2774749891u32,        3873578179u32,  3554797811u32,  4248601033u32,  1954127977u32,  3696687607u32,         424225451u32,   723102988u32,  2727644884u32,    80131879u32,  3436989742u32,        2359556822u32,   761867225u32,  1998925892u32,  3206342622u32,  3009214665u32,        1116804721u32,    62006803u32,   973179987u32,  2054767372u32,  2824152325u32,        2980313218u32,  4277895641u32,  3846084523u32,   351878265u32,  1804199144u32,           7817541u32,  3399899219u32,   235305462u32,  1925961770u32,  1732496705u32,         107650687u32,  1557009902u32,  1621660041u32,  2482640488u32,   111666002u32,        2083676578u32,   783315473u32,    56145210u32,  2664751553u32,  1159555293u32,        2627443834u32,   720794756u32,  4261634422u32,   239242866u32,  2818226587u32,        3888434822u32,    20448685u32,  2596166878u32,   868753769u32,  3639375934u32,        2241017830u32,   959123451u32,  1713858054u32,   787845463u32,  3022593980u32,        3590234110u32,  1928949400u32,  1974227998u32,  1476271115u32,  3617068776u32,          77989191u32,  2337224653u32,  1069854417u32,   531551502u32,  1573487592u32,        1075162631u32,  3701391979u32,   391745609u32,  3247974457u32,  3233749597u32,        1939323778u32,  1105178271u32,  3536028962u32,  3400208985u32,  2585100896u32,         803157615u32,  4241569712u32,  2231721671u32,  2579043372u32,  4137423228u32,        2365490527u32,   493858877u32,   713290230u32,  1184962289u32,  3356493105u32,        3793338427u32,   471574670u32,  3521825386u32,  3614958614u32,   150033603u32,         956370420u32,  3130540136u32,  1550758543u32,  4042530294u32,   391073114u32,        2689793138u32,   291816517u32,  3671286337u32,  3169434396u32,  1289477195u32,        2136498169u32,  3753771740u32,  2257438616u32,  1503287642u32,   650373666u32,        4138387416u32,  4042568065u32,  2203828331u32,   728074500u32,  3438525617u32,         737340775u32,  3257981776u32,  1016252979u32,    36247035u32,   968355266u32,        2625570223u32,     1279026u32,   496220977u32,  1651478522u32]);
    panic!("{:?}", mt.next_u32());

    let mut hyperedge_indices                           = Vec::with_capacity(((difficulty.num_nodes * difficulty.num_edges)+1) / difficulty.num_edges + 1);
    for i in (0..(difficulty.num_nodes * difficulty.num_edges)+1).step_by(difficulty.num_edges)
    {
        hyperedge_indices.push(i as u64);
    }

    let vertices                                        : Vec<u64> = (0..difficulty.num_vertices as u64).collect();

    let mut hyperedges                                  = Vec::with_capacity(difficulty.num_nodes);
    for i in 0..difficulty.num_nodes
    {
        let mut vec                                     = Vec::with_capacity(difficulty.num_edges);
        for j in 0..difficulty.num_edges
        {
            vec.push(
                vertices[(rng.next_u32()%difficulty.num_vertices as u32) as usize]
            );
        }

        hyperedges.push(vec);
    }

    return (vertices, hyperedge_indices, hyperedges);
}

impl crate::ChallengeTrait<Solution, Difficulty, 4> for Challenge 
{
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance(
        seed:                           [u8; 32],
        difficulty:                     &Difficulty,
        dev:                            &Arc<CudaDevice>,
        mut funcs:                      HashMap<&'static str, CudaFunction>,
    )                                               -> Result<Self> 
    {
        // TIG dev bounty available for a GPU optimisation for instance generation!
        return Self::generate_instance(seed, difficulty);
    }

    fn generate_instance(
        seed:                           [u8; 32], 
        difficulty:                     &Difficulty
    )                                               -> Result<Challenge> 
    {
        let (vertices, hyperedge_indices, hyperedges)   = get_init_values(difficulty, seed);

        let result                                      = solve_greedy_bipartition(&vertices, &hyperedges, Some(difficulty.num_blocks));

        return Ok(Challenge
        {
            seed:                                       seed,
            difficulty:                                 difficulty.clone(),
            vertices:                                   vertices,
            hyperedge_indices:                          hyperedge_indices,
            hyperedges:                                 hyperedges,
            node_weights:                               vec![1.0f32; difficulty.num_vertices as usize],
            edge_weights:                               vec![1.0f32; difficulty.num_edges as usize],
        });
    }

    fn verify_solution(&self, solution: &Solution)  -> Result<()> 
    {
        return Ok(());
    }
}

fn recursive_bipartition(
    vertices_subset:                    &Vec<u64>, 
    partitions:                         &mut Vec<i32>,
    vertex_to_hyperedges:               &Vec<Vec<usize>>,
    current_id:                         i32, 
    current_depth:                      u32, 
    partitions_per_subset:              usize
)
{
    if current_depth == 0
    {
        for i in 0..vertices_subset.len()
        {
            partitions[i]                               = current_id;
        }

        return;
    }

    let half_partitions                                 = partitions_per_subset / 2;
    let left_partitions                                 = half_partitions;
    let right_partitions                                = partitions_per_subset - half_partitions;

    let target_left                                     = vertices_subset.len() * left_partitions / partitions_per_subset;
    let target_right                                    = vertices_subset.len() - target_left;

    let (left, right)                                   = bipartition(vertices_subset, vertex_to_hyperedges, target_left, target_right);

    recursive_bipartition(&left, partitions, vertex_to_hyperedges, current_id * 2, current_depth - 1, left_partitions);
    recursive_bipartition(&right, partitions, vertex_to_hyperedges, current_id * 2 + 1, current_depth - 1, right_partitions);
}

fn bipartition(
    vertices_subset:                    &Vec<u64>,
    vertex_to_hyperedges:               &Vec<Vec<usize>>,
    target_left:                        usize, 
    target_right:                       usize
)                                                   -> (Vec<u64>, Vec<u64>)
{
    assert!(target_left + target_right == vertices_subset.len());

    let mut degrees                                     = Vec::<usize>::with_capacity(vertices_subset.len());
    for v in vertices_subset
    {
        degrees.push(vertex_to_hyperedges[*v as usize].len());
    }

    let mut sorted_vertices                             = vertices_subset.clone();
    sorted_vertices.sort_by_key(|&i| degrees[i as usize]);

    panic!("{:?}", sorted_vertices);

    return (Vec::new(), Vec::new());
}

fn solve_shape(
    hyperedges:                         &Vec<Vec<u64>>
)                                                   -> (usize, usize)
{
    return (hyperedges.len(), hyperedges[0].len());
}

fn solve_greedy_bipartition(
    vertices:                           &Vec<u64>, 
    hyperedges:                         &Vec<Vec<u64>>,
    num_partitions:                     Option<usize>
)                                                   -> Vec<i32>
{
    let depth                                           = num_partitions.unwrap_or(16).ilog2();
    let M                                               = solve_shape(hyperedges).0;

    // Preprocessing: Build mappings
    // vertex_to_hyperedges[v] will contain the hyperedge indices that include vertex v
    let mut vertex_to_hyperedges                        : Vec<Vec<usize>> = vec![Vec::with_capacity(hyperedges.len()); vertices.len()];
    for i in 0..hyperedges.len()
    {
        for j in hyperedges[i].iter()
        {
            vertex_to_hyperedges[*j as usize].push(i);
        }
    }

    let mut partitions                                  = vec![-1 as i32; vertices.len()];
    recursive_bipartition(vertices, &mut partitions, &vertex_to_hyperedges, 0, depth, num_partitions.unwrap_or(16));

    return partitions;
}