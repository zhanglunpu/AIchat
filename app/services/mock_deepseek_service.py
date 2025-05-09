class MockDeepseekService:
    """
    Deepseek模型的模拟服务，用于开发和测试
    实际应用中应该使用真正的Deepseek API
    """
    def __init__(self):
        # 简单的知识库，用于模拟回答
        self.knowledge_base = {
            "黄芪": "黄芪，性味甘、微温，归肺、脾、肝、肾经。主要功效是补气升阳、益卫固表、利水消肿、生津养血、行滞通痹等。常用于气虚乏力、食少便溏、中气下陷、久泻脱肛、表虚自汗、气虚水肿、内热消渴等症。现代研究表明，黄芪具有增强免疫力、抗氧化、抗衰老、保护心血管等作用。禁忌：表实邪盛、热毒炽盛者慎用，脾胃有湿热者不宜。",
            "枸杞": "枸杞子，性味甘、平，归肝、肾、肺经。具有滋补肝肾、益精明目、补血安神、生津止渴等功效。主治肝肾阴虚、精血不足、腰膝酸软、头晕耳鸣、目视昏花、内热消渴等症。现代研究表明，枸杞含有丰富的枸杞多糖、胡萝卜素、维生素等，具有抗氧化、增强免疫力、保护肝脏等作用。食用禁忌：脾胃虚寒、腹泻者慎用；风热感冒、实热症状者忌用。",
            "当归": "当归，性味甘、辛、温，归肝、心、脾经。主要功效是补血活血、调经止痛、润肠通便。常用于血虚萎黄、月经不调、经闭痛经、虚寒腹痛、肠燥便秘、风湿痹痛、跌打损伤等症。现代研究表明，当归具有抗凝血、抗血栓、改善微循环等作用。禁忌：体内有实热、湿热者慎用；月经过多者不宜服用；孕妇慎用。",
            "金银花": "金银花，性味甘、寒，归肺、心、胃经。主要功效是清热解毒、疏散风热。常用于热毒血痢、痈肿疮毒、温病发热、风热感冒等症。现代研究表明，金银花具有广谱抗菌、抗病毒、抗炎等作用。禁忌：阴虚内热、脾胃虚寒者慎用。",
            "槐花": "槐花，性味苦、微寒，归肝、大肠经。主要功效是清热凉血、止血。常用于血热出血证如便血、痔血、血痢、崩漏等。现代研究表明，槐花含有芦丁等黄酮类化合物，具有抗氧化、抗炎、保护血管等作用。禁忌：气虚、阳虚者慎用；孕妇慎用。"
        }
    
    async def generate_response(self, prompt: str) -> str:
        """
        模拟生成回答
        """
        # 检查是否包含知识库中的关键词
        for herb, info in self.knowledge_base.items():
            if herb in prompt:
                return self._format_response(herb, info)
        
        # 通用回答
        return self._generate_generic_response(prompt)
    
    def _format_response(self, herb: str, info: str) -> str:
        """格式化中药回答"""
        return f"关于{herb}的信息如下：\n\n{info}\n\n以上是{herb}的基本信息，如果您有更具体的问题，请继续咨询。"
    
    def _generate_generic_response(self, prompt: str) -> str:
        """生成通用回答"""
        if "配伍" in prompt or "搭配" in prompt:
            return "中药配伍是中医用药的重要原则，主要包括七情配伍：单行、相须、相使、相畏、相杀、相恶、相反。合理的配伍可以增强疗效、减少毒副作用。不同药材搭配需要考虑各自的性味、归经和功效，确保协同作用。如果您想了解具体的药材配伍，请提供药材名称，我可以为您提供更详细的信息。"
        
        elif "禁忌" in prompt:
            return "中药使用禁忌一般包括：1. 妊娠禁忌：如巴豆、牵牛子等破血逐瘀药物孕妇禁用；2. 体质禁忌：如阴虚体质者慎用温燥药物；3. 疾病禁忌：如高血压患者慎用兴奋类药物；4. 配伍禁忌：如十八反、十九畏等。具体药材的禁忌因药而异，如果您关心特定药材，请告诉我药材名称。"
        
        elif "用量" in prompt or "怎么吃" in prompt:
            return "中药用量需要遵循中医辨证施治原则，根据个人体质、病情轻重、年龄大小等因素综合考量。一般来说，成人常用量见于药典，儿童、老人、体弱者适当减量。煎煮时间通常为20-30分钟，每日1-2剂。但自行用药存在风险，建议在中医师指导下使用。如需具体药材的用法用量，请提供药材名称。"
        
        else:
            return "作为中医药顾问，我可以回答您关于中草药的功效、性味、归经、配伍、禁忌等问题。请提供具体的中药名称或者明确的中医药相关问题，我会为您提供专业的解答。" 