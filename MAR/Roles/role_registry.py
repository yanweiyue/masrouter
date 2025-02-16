import json

class RoleRegistry:
    def __init__(self, domain, role):
        self.domain = domain
        self.role = role
        self.role_profile = self.get_role_profile()
    
    def get_role_profile(self):
        profile = json.load(open(f"MAR/Roles/{self.domain}/{self.role}.json"))
        return profile
    
    def get_name(self):
        return self.role_profile['Name']
    
    def get_message_aggregation(self):
        return self.role_profile['MessageAggregation']
    
    def get_description(self):
        return self.role_profile['Description']
    
    def get_output_format(self):
        return self.role_profile['OutputFormat']
    
    def get_reasoning(self):
        return self.role_profile['Reasoning']
    
    def get_post_process(self):
        return self.role_profile['PostProcess']
    
    def get_post_description(self):
        return self.role_profile['PostDescription']
    
    def get_post_output_format(self):
        return self.role_profile['PostOutputFormat']
    