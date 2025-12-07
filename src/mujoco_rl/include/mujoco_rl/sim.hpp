#include <memory>       
#include <unordered_map>
#include <string>  

class Sim {
public:
    Sim::Sim();
    void set_ctrl(std::shared_ptr<std::unordered_map<std::string, double>> ctrl_ptr);
    void create_model(std::string model_path);
private:
    std::shared_ptr<std::unordered_map<std::string, double>> ctrl_ptr;

};