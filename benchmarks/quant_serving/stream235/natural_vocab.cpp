#include "llama.h"
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
// No inference context and no weights. Load only the tokenizer metadata view.
int main(int argc, char**argv){if(argc!=3)return 2;Dl_info info{};if(!dladdr(reinterpret_cast<void*>(&llama_tokenize),&info) || !std::filesystem::equivalent(info.dli_fname,argv[2]))return 6;auto params=llama_model_default_params();params.vocab_only=true;params.n_gpu_layers=0;params.use_mmap=false;
    auto*model=llama_model_load_from_file(argv[1],params);if(!model)return 3;auto*v=llama_model_get_vocab(model);int n=llama_vocab_n_tokens(v);std::cout<<"vocab "<<n<<"\neog";for(int i=0;i<n;i++)if(llama_vocab_is_eog(v,i))std::cout<<" "<<i;std::cout<<"\n";
    std::string text;while(std::getline(std::cin,text)){ // Input is hex UTF-8, including the newline used by real prompts.
        if(text.size()<3 || text[1]!=':' || (text[0]!='0' && text[0]!='1'))return 7;bool special=(text[0]=='1');text.erase(0,2);
        if(text.size()%2 || text.size()>262144 || text.find_first_not_of("0123456789abcdef")!=std::string::npos)return 4;
        std::string in;for(size_t i=0;i<text.size();i+=2)in.push_back(char(std::stoi(text.substr(i,2),nullptr,16)));
        std::vector<llama_token> ids(in.size()+16);int got=llama_tokenize(v,in.data(),int(in.size()),ids.data(),int(ids.size()),false,special);if(got<0)return 5;
        std::cout<<"tokens";for(int i=0;i<got;i++)std::cout<<" "<<ids[i];std::cout<<"\n";}
    llama_model_free(model);return 0;
}
