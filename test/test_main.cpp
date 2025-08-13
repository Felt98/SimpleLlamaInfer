#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  
  // 确保日志目录存在
  mkdir("./log", 0755);
  
  google::InitGoogleLogging("SimpleLlamaInfer");
  FLAGS_log_dir = "./log/";
  FLAGS_alsologtostderr = true;

  LOG(INFO) << "Start Test...\n";
  return RUN_ALL_TESTS();
}