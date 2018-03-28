//
//  ViewController.m
//  TFMotionDetector
//
//  Created by Hemant Sachdeva on 26/03/18.
//  Copyright © 2018 Hemant Sachdeva. All rights reserved.
//

#import "ViewController.h"
#import <CoreMotion/CoreMotion.h>
#import <tensorflow/core/public/session.h>

@interface ViewController ()
@property (nonatomic, strong) UIView * ball;
@property (strong, nonatomic) CMMotionManager *motionManager;
@property bool isModelLoaded;
@property float updateInterval;
@property (weak, nonatomic) IBOutlet UILabel *lblIsMoving;
@property (weak, nonatomic) IBOutlet UILabel *lblUserAcc;
@property (weak, nonatomic) IBOutlet UILabel *lblFilteredValue;
@property (weak, nonatomic) IBOutlet UILabel *lblSigmoidValue;
@end

@implementation ViewController
{
    NSTimer* mainTimer;
    float *inState;
    CMAcceleration userA;
    tensorflow::Session *session;
    
}
float R = 16;


- (void)initBall
{
    self.ball = [[UIView alloc] initWithFrame:CGRectMake(160, 250, R, R)];
    self.ball.layer.cornerRadius = 8;
    self.ball.backgroundColor = [UIColor blueColor];
    [self.view addSubview:self.ball];
}
- (void) viewDidDisappear:(BOOL)animated{
    NSLog(@"called");
    [mainTimer invalidate];
}
- (void)viewDidLoad {
    [super viewDidLoad];
    [self initBall];
    self.updateInterval = 1.0/100.0;
    self.ball.frame = CGRectMake(self.lblIsMoving.frame.origin.x, self.lblIsMoving.frame.origin.y, R, R);
    self.isModelLoaded = NO;
    mainTimer = [NSTimer scheduledTimerWithTimeInterval:self.updateInterval  target:self selector:@selector(motionRefresh:) userInfo:nil repeats:NO];
    
    
    self.motionManager = [[CMMotionManager alloc] init];
    
    self.motionManager.accelerometerUpdateInterval = self.updateInterval ;
    [self.motionManager startDeviceMotionUpdatesUsingReferenceFrame:CMAttitudeReferenceFrameXArbitraryZVertical];
}

- (void)motionRefresh:(id)sender {
    if(self.motionManager == nil){
        mainTimer = [NSTimer scheduledTimerWithTimeInterval:self.updateInterval  target:self selector:@selector(motionRefresh:) userInfo:nil repeats:NO];
        return;
    }
    userA = self.motionManager.deviceMotion.userAcceleration;
    NSLog(@"Ax is %f", userA.x);
    NSLog(@"Ay is %f", userA.y);
    NSLog(@"Az is %f", userA.z);
    float g = 9.8;

    if(!self.isModelLoaded){
        [self loadModels];
    }else {
        userA.x = userA.x * g;
        userA.y = userA.y * g;
        userA.z = userA.z * g;
        [self showResults];
    }
    mainTimer = [NSTimer scheduledTimerWithTimeInterval:self.updateInterval  target:self selector:@selector(motionRefresh:) userInfo:nil repeats:NO];
}

- (void) showResults {
    /*
     For user acceleration
     */
    
    tensorflow::Tensor ax(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    ax.scalar<float>()() = userA.x;
    
    tensorflow::Tensor ay(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    ay.scalar<float>()() = userA.y;
    
    tensorflow::Tensor az(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    az.scalar<float>()() = userA.z;

    /*
     For motion dectection filter value
     */
    if (inState == nil){
        inState = new float[3];
        inState[0] = 0.0f;
        inState[1] = 0.0f;
        inState[2] = 0.0f;
    }
    tensorflow::Tensor inStateT(tensorflow::DT_FLOAT, tensorflow::TensorShape({3,}));
    inStateT.vec<float>()(0) = inState[0];
    inStateT.vec<float>()(1) = inState[1];
    inStateT.vec<float>()(2) = inState[2];

    // The feed dictionary for doing inference.
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {"mf/ax", ax},
        {"mf/ay", ay},
        {"mf/az", az},
        {"mf/iir/InState", inStateT},

    };
    // We want to run these nodes.
    std::vector<std::string> nodes = {
        {"mf/a"},
        {"mf/iir/Y"},
        {"mf/iir/OutState"},
        {"mf/y"},
        
    };

    // The results of running the nodes are stored in this vector.
    std::vector<tensorflow::Tensor> outputs;
    
    // Run the session.
    auto status = session->Run(inputs, nodes, {}, &outputs);
    if (!status.ok()) {
        NSLog(@"Error running model: %s", status.error_message().c_str());
    }
    
    
    auto linearAcc = outputs[0].tensor<float, 0>();
    auto filteredLa = outputs[1].tensor<float, 0>();
    auto mfOutState = outputs[2].tensor<float, 1>();
    auto isMoving = outputs[3].tensor<float, 0>();
    
    inState[0] = mfOutState(0);
    inState[1] = mfOutState(1);
    inState[2] = mfOutState(2);

    
    self.ball.backgroundColor = [UIColor blueColor];
    if( isMoving(0) >= 0.5){
        self.ball.backgroundColor = [UIColor yellowColor];
    }
    
    [ _lblUserAcc setText: [NSString stringWithFormat:@"x: % 04.2f,\ty: % 04.2f,\tz: % 04.2f,\ta.a: % 04.2f", userA.x,  userA.y,  userA.z, linearAcc(0)]];
    [_lblFilteredValue setText:[NSString stringWithFormat:@"%.05f", filteredLa(0)]];
    [_lblSigmoidValue setText:[NSString stringWithFormat:@"%.05f", isMoving(0)]];
}

- (void) loadModels{
    NSString *md = [[NSString alloc]initWithString:[[NSBundle mainBundle] pathForResource: @"motion_detector" ofType:@"pb"]];
    if ([self loadGraphsFromPath: [NSArray arrayWithObjects:md,  nil]]){
            self.isModelLoaded = YES;
    }
        
}

- (BOOL)createSession:(tensorflow::GraphDef) graph
{
    tensorflow::SessionOptions options;
    options.config.set_inter_op_parallelism_threads(1);
    options.config.set_intra_op_parallelism_threads(1);
    
    auto status = tensorflow::NewSession(options, &session);
    if (!status.ok()) {
        NSLog(@"Error creating session: %s", status.error_message().c_str());
        return NO;
    }
    
    status = session->Create(graph);
    if (!status.ok()) {
        NSLog(@"Error adding graph to session: %s", status.error_message().c_str());
        return NO;
    }
    
    return YES;
}

- (BOOL)loadGraphsFromPath:(NSArray<NSString*> *)paths
{
    BOOL sessionCreated = NO;
    for(NSString* path in paths){
        tensorflow::GraphDef graph;
        auto status = ReadBinaryProto(tensorflow::Env::Default(), path.fileSystemRepresentation, &graph);
        if (!status.ok()) {
            NSLog(@"Error reading graph: %s", status.error_message().c_str());
            return NO;
        }
        
        auto nodeCount = graph.node_size();
        NSLog(@"Node count: %d", nodeCount);
        for (auto i = 0; i < nodeCount; ++i) {
            auto node = graph.node(i);
            //NSLog(@"Node %d: %s '%s'", i, node.op().c_str(), node.name().c_str());
        }
        
        if(sessionCreated == NO){
            [self createSession:graph];
            sessionCreated = YES;
        }else{
            status = session->Extend(graph);
            if (!status.ok()) {
                NSLog(@"Error extending graph to session: %s", status.error_message().c_str());
                return NO;
            }
            
        }
    }
    return YES;
}
    
- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
