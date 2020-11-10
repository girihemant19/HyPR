#include "json/json.h"
#include <iostream>
using namespace std;

using namespace std;
Json::Reader reader;
Json::Value value;

int main()
{
    string strValue = "{\"name\":\"json\",\"Year\":1994,\"Sex\":\"Man\",\"others\":{\"Height\":180,\"Weight\":150},\"Parents\":[{\"Father\":\"C++\"},{\"Mother\":\"C\"}]}";
    cout<<strValue<<endl;
    if (reader.parse(strValue, value))
    {
        Json::FastWriter fast_writer;
        cout << fast_writer.write(value) <<endl;
        
        string out = value["name"].asString();
        cout<<out <<endl;
        out = value["Year"].asString();
        cout<<out <<endl;
        out = value["Sex"].asString();
        cout<<out <<endl;
        out = value["others"]["Height"].asString();
        cout<<out <<endl;
        out = value["others"]["Weight"].asString();
        cout<<out <<endl;
        const Json::Value arrayObj = value["Parents"];
        for (unsigned int i = 0; i < arrayObj.size(); i++)
        {
            if (arrayObj[i].isMember("Father"))
            {
                out = arrayObj[i]["Father"].asString();
                cout<<out <<endl;
            }
            if (arrayObj[i].isMember("Mother"))
            {
                out = arrayObj[i]["Mother"].asString();
                cout<<out <<endl;
            }
        }
    }
    
    
    Json::Value json_temp;
    json_temp["Height"] = Json::Value(180.2);
    json_temp["Weight"] = Json::Value(150);
    
    Json::Value Parents,Mother,Father;
    Father["Father"] = Json::Value("C++");
    Mother["Mother"] = Json::Value("C");
    Parents.append(Father);
    Parents.append(Mother);
    
    Json::Value root;
    root["name"] = Json::Value("json");
    root["Year"] = Json::Value(1994);
    root["Sex"] = Json::Value("Man");
    root["others"] = json_temp;
    root["Parents"] = Parents;
    Json::FastWriter fast_writer;
    cout << fast_writer.write(root) <<endl;
    
    
    
    return 0;
}
