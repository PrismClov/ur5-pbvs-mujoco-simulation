// Single-marker observation/motion adapter; upstream EKF implementation is unchanged.
#include "tools/extended_kalman_filter.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;
using T4 = Eigen::Matrix<double,4,4,Eigen::RowMajor>;
namespace {
double wrap(double a) { return std::atan2(std::sin(a),std::cos(a)); }
struct Filter {
  tools::ExtendedKalmanFilter ekf;
  bool initialized=false;
  double time=0, nis=0, linear_psd, angular_psd, pos_std, angle_std, gap, speed, omega;
  int count=0;
  Filter(double a,double b,double p,double r,double g,double v,double w)
    : linear_psd(a),angular_psd(b),pos_std(p),angle_std(r),gap(g),speed(v),omega(w) {
    for(double value : {a,b,p,r,g,v,w})
      if(!std::isfinite(value)||value<=0) throw std::invalid_argument("invalid config");
  }
  tools::ExtendedKalmanFilter future(double t) const {
    if(!initialized || !std::isfinite(t) || t<time || t-time>.5)
      throw std::invalid_argument("invalid prediction time");
    double dt=t-time;
    M F=M::Identity(8,8),Q=M::Zero(8,8);
    F.block<3,3>(0,3)=Eigen::Matrix3d::Identity()*dt;
    F(6,7)=dt;
    for(int i=0;i<4;i++) {
      int p=i<3?i:6, v=i<3?i+3:7;
      double s=i<3?linear_psd:angular_psd;
      Q(p,p)=s*dt*dt*dt/3; Q(p,v)=Q(v,p)=s*dt*dt/2; Q(v,v)=s*dt;
    }
    auto result=ekf;
    result.predict(F,Q,[F](const V& x)->V { V y=F*x;y[6]=wrap(y[6]);return y; });
    return result;
  }
  static V h(const V& x,const T4& camera) {
    Eigen::Vector3d p=camera.block<3,3>(0,0).transpose()*(x.head<3>()-camera.block<3,1>(0,3));
    if(p.z()<=.03) throw std::invalid_argument("invalid depth");
    V z(4);z<<p.x()/p.z(),p.y()/p.z(),p.z(),x[6];return z;
  }
  int update(double t,const T4& camera,const T4& marker) {
    if(!std::isfinite(t)||!camera.allFinite()||!marker.allFinite()||marker(2,3)<=.03)
      return -1;
    if(initialized && t<=time) return -1;
    T4 world=camera*marker;
    if(std::hypot(world(0,0),world(1,0))<.05) return -1; // yaw is undefined here
    double yaw=std::atan2(world(1,0),world(0,0));
    if(!initialized || t-time>gap) {
      V x=V::Zero(8);x.head<3>()=world.block<3,1>(0,3);x[6]=yaw;
      V variance(8);variance<<pos_std*pos_std,pos_std*pos_std,pos_std*pos_std,
        .25,.25,.25,angle_std*angle_std,.25;
      ekf=tools::ExtendedKalmanFilter(x,variance.asDiagonal(),[](const V&a,const V&b)->V {
        V result=a+b; result[6]=wrap(result[6]);return result;
      });
      initialized=true; time=t;count=1;nis=0;return 1;
    }
    auto candidate=future(t);
    const auto& x=candidate.x;
    auto Rt=camera.block<3,3>(0,0).transpose().eval();
    Eigen::Vector3d p=Rt*(x.head<3>()-camera.block<3,1>(0,3));
    if(p.z()<=.03) return 0;
    double z=p.z();
    Eigen::Matrix3d D;
    D<<1/z,0,-p.x()/(z*z),0,1/z,-p.y()/(z*z),0,0,1;
    M H=M::Zero(4,8);H.block<3,3>(0,0)=D*Rt;H(3,6)=1;
    // Four residual entries satisfy the original EKF diagnostic layout.
    V measurement(4);measurement<<marker(0,3)/marker(2,3),marker(1,3)/marker(2,3),marker(2,3),yaw;
    double bearing=pos_std/marker(2,3);
    V variances(4);variances<<bearing*bearing,bearing*bearing,pos_std*pos_std,angle_std*angle_std;
    M R=variances.asDiagonal();
    auto subtract=[](const V&a,const V&b)->V {V d=a-b;d[3]=wrap(d[3]);return d;};
    V residual=subtract(measurement,h(x,camera));
    M S=H*candidate.P*H.transpose()+R;
    nis=residual.dot(S.ldlt().solve(residual));
    if(!std::isfinite(nis)||nis>18.47) return 0; // independent PRIOR gate, df=4
    candidate.update(measurement,H,R,[camera](const V& state){return h(state,camera);},subtract);
    if(!candidate.x.allFinite()||!candidate.P.allFinite()||candidate.x.segment<3>(3).norm()>speed
       ||std::abs(candidate.x[7])>omega) return 0;
    ekf=std::move(candidate);time=t;count++;return 2;
  }
};
}

API void* tj_create(double a,double b,double p,double r,double g,double v,double w) {
  try {return new Filter(a,b,p,r,g,v,w);} catch(...) {return nullptr;}
}
API void tj_destroy(void* ptr) { delete static_cast<Filter*>(ptr); }
API int tj_update(void* ptr,double t,const double* camera,const double* marker) {
  if(!ptr||!camera||!marker)return -1;
  try {return static_cast<Filter*>(ptr)->update(t,Eigen::Map<const T4>(camera),Eigen::Map<const T4>(marker));}
  catch(...) {return -1;}
}
API int tj_predict(void* ptr,double t,double* state,double* covariance,double* nis,int* count) {
  if(!ptr||!state||!covariance||!nis||!count)return -1;
  try {
    auto& f=*static_cast<Filter*>(ptr);auto result=f.future(t);
    Eigen::Map<V> output_state(state,8); output_state=result.x;
    Eigen::Map<Eigen::Matrix<double,8,8,Eigen::RowMajor>> output_covariance(covariance);
    output_covariance=result.P;
    *nis=f.nis;*count=f.count;return 0;
  } catch(...) {return -1;}
}
