from fastapi import FastAPI, HTTPException, UploadFile, File
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
from pydantic import BaseModel
from typing import Dict, Any


import functions

# 定义全局变量
global_job_title = None
global_resume_text = None
global_initial_results = None
global_all_rate_results = None
global_personalization = None
global_polish_suggestions = None 
global_memory_dict = None
global_current_index = '1'
global_desc_polished_all_project = None

app = FastAPI(
    title="简历分析API",
    description="一个使用FastAPI构建的简历分析API，提供简历解析和多维度评估功能。",
    version="0.1.0",
)

# 配置 CORS 中间件，允许所有来源的跨域请求（在生产环境中应更严格）
# 暂时允许所有来源，部署后再限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 暂时允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_resume/", 
            summary="上传PDF简历并进行完整分析",
            description="接收PDF文件和目标职位名称，提取简历文本，并行运行初步分析和所有评分函数，并返回所有结果。")
async def analyze_resume_full_process(job_title: str, resume_file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    处理PDF简历上传，提取文本，结合职位名称进行初步分析，然后进行所有评分分析。

    - **job_title**: 用户输入的目标职位名称 (作为表单字段或查询参数)。
    - **resume_file**: 用户上传的PDF简历文件。

    返回一个包含提取的简历文本、初步分析结果和所有评分结果的字典。
    """
    global global_job_title, global_resume_text, global_initial_results, global_all_rate_results
    
    temp_pdf_path = None
    try:
        # 1. 保存上传的PDF文件到临时位置
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        # 确保文件名安全，防止路径遍历等问题
        safe_filename = os.path.basename(resume_file.filename)
        temp_pdf_path = os.path.join(temp_dir, safe_filename)
        
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(resume_file.file, buffer)

        # 2. 在线程池中调用 functions.py 中的 get_resume_and_initial_results_parallel
        #    这个函数接收 (pdf_path, job_title) 并返回 (resume_text, initial_analysis_results)
        resume_text, initial_analysis_results = await asyncio.to_thread(functions.get_resume_and_initial_results_parallel, temp_pdf_path, job_title)

        if "Error extracting text" in resume_text or not resume_text: # 确保简历文本有效
            raise HTTPException(status_code=400, detail=f"简历解析失败或内容为空: {resume_text}")
        
        # 3. 在线程池中调用 functions.py 中的 run_all_rate_functions_parallel
        #    它需要 job_title, resume_text, 和 initial_analysis_results
        all_rate_results = await asyncio.to_thread(
            functions.run_all_rate_functions_parallel,
            job=job_title, # 使用从请求中获取的 job_title
            resume=resume_text, # 使用上一步提取的 resume_text
            results=initial_analysis_results # 使用上一步获取的 initial_analysis_results
        )
        
        # 保存到全局变量
        global_job_title = job_title
        global_resume_text = resume_text
        global_initial_results = initial_analysis_results
        global_all_rate_results = all_rate_results
        
        response_data = {
            "resume_text": resume_text,  # 添加这行
            "initial_analysis": initial_analysis_results,
            "all_ratings": all_rate_results
        }
        return response_data

    except HTTPException as http_exc: # 重新抛出已知的HTTP异常
        raise http_exc
    except Exception as e:
        # 对于其他未知错误，记录日志可能更有帮助
        # import logging
        # logging.exception("Error during full resume analysis")
        raise HTTPException(status_code=500, detail=f"处理完整简历分析请求时发生内部错误: {str(e)}")
    finally:
        # 清理临时文件
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                # logging.error(f"Failed to remove temp file {temp_pdf_path}: {e}")
                pass # 或者记录错误
        if resume_file:
            await resume_file.close()





# 添加请求模型
# 修改请求模型，使其他字段可选
class PolishRequest(BaseModel):
    personalization: Dict[str, str]
    job_title: str = ""
    resume_text: str = ""
    initial_results: Dict[str, Any] = {}
    all_rate_results: Dict[str, Any] = {}

# 在现有代码后添加新的API端点
@app.post("/polish_resume/", 
            summary="简历优化建议",
            description="根据分析结果和个人偏好设置，生成简历优化建议。")
async def polish_resume_endpoint(request: PolishRequest) -> Dict[str, Any]:
    """
    根据分析结果和个人偏好设置，生成简历优化建议。
    """
    global global_job_title, global_resume_text, global_initial_results, global_all_rate_results, global_personalization, global_polish_suggestions
    global global_current_index
    global_current_index = '1'
    try:
        # 保存个性化设置到全局变量
        global_personalization = request.personalization
        
        # 优先使用全局变量，如果全局变量为空则使用请求中的数据
        job_title = global_job_title or request.job_title
        resume_text = global_resume_text or request.resume_text
        initial_results = global_initial_results or request.initial_results
        all_rate_results = global_all_rate_results or request.all_rate_results
        
        print(job_title)
        print(resume_text)
        print(initial_results)
        print(all_rate_results)
        print(request.personalization)

        # 在线程池中调用 polish_resume 函数
        polish_suggestions = await asyncio.to_thread(
            functions.polish_resume,
            job_title,
            resume_text,
            initial_results,
            all_rate_results,
            request.personalization
        )
        
        # 保存简历优化建议到全局变量
        global_polish_suggestions = polish_suggestions
        
        return {
            "polish_suggestions": polish_suggestions,
            "status": "success"
        }
    except Exception as e:
        print(f"Polish resume error: {str(e)}")  # 添加调试信息
        raise HTTPException(status_code=500, detail=f"生成简历优化建议时发生错误: {str(e)}")

class ChatFeedbackRequest(BaseModel):
    user_message: str
    current_marker: str = ""

@app.post("/chat_feedback/", 
            summary="处理用户聊天反馈",
            description="接收用户在第三张卡片中的聊天反馈，处理后返回更新的建议。")
async def chat_feedback(request: ChatFeedbackRequest) -> Dict[str, Any]:
    try:
        global global_polish_suggestions, global_job_title, global_resume_text, global_initial_results, global_all_rate_results, global_personalization


        original_polish_suggestions = global_polish_suggestions
        job_title = global_job_title
        resume_text = global_resume_text
        initial_results = global_initial_results
        all_rate_results = global_all_rate_results
        personalization = global_personalization
        user_message = request.user_message
        current_marker = request.current_marker

    
        updated_suggestions = await asyncio.to_thread(
            functions.update_polish_suggestions,
            user_message, 
            current_marker,
            original_polish_suggestions,
            job_title,
            resume_text,
            initial_results,
            all_rate_results,
            personalization
        )        

        
        global_polish_suggestions = updated_suggestions

        return {
            'success': True,
            'updated_suggestions': updated_suggestions,
            'message': '反馈处理成功'
        }
        
    except Exception as e:
        print(f"处理聊天反馈时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理聊天反馈时发生错误: {str(e)}")

class StartOptimizationRequest(BaseModel):
    confirmation: bool = True

# 在chat_feedback函数后面添加新的API端点
@app.post("/start_optimization/", 
            summary="开始具体优化",
            description="处理用户确认所有方案后的具体优化请求。")
async def start_optimization(request: StartOptimizationRequest) -> Dict[str, Any]:
    """
    处理用户确认所有方案后的具体优化请求。
    目前返回一个示例字典，后续可以添加具体的优化逻辑。
    """
    try:
        global global_job_title, global_resume_text, global_initial_results, global_all_rate_results, global_personalization, global_polish_suggestions, global_memory_dict, global_desc_polished_all_project
        polish_suggestions = global_polish_suggestions
        job_title = global_job_title
        resume_text = global_resume_text
        initial_results = global_initial_results
        all_rate_results = global_all_rate_results
        personalization = global_personalization

        memory_dict = functions.create_memory(polish_suggestions)


        desc_polished_all_project = await asyncio.to_thread(
            functions.polishing_all_project,
            memory_dict, 
            job_title, 
            resume_text, 
            initial_results, 
            all_rate_results, 
            personalization
        )

        
        # 将polished_all_project的结果添加到memory_dict中
        for index, desc_polished_project in desc_polished_all_project.items():
            polished_project = functions.extract_content_between_dashes(desc_polished_project)
            memory_dict = functions.add_memory(index, memory_dict,user_message = "请帮我具体优化指定的项目经历，按照格式给出修改后的新版项目经历描述", AI_message = desc_polished_project, polished_project=polished_project)
        global_memory_dict = memory_dict
        current_index = '1'
        global_desc_polished_all_project = desc_polished_all_project
        header = functions.extract_header(polish_suggestions, current_index)

        results = {
            "success" : True,
            "current_index": '1',
            'header':header,
            "memory_dict": memory_dict,
            "desc_polished_all_project": desc_polished_all_project,
            "message": "处理成功"

        }        
        return results
        
    except Exception as e:
        print(f"开始优化时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"开始优化时发生错误: {str(e)}")


# 在现有的API端点后添加新的端点
# 修改请求模型
class NextItemRequest(BaseModel):
    action: str = "next"
    center_content: str = ""  # 添加中间容器内容字段

class PreviousItemRequest(BaseModel):
    action: str = "previous"
    center_content: str = ""  # 添加中间容器内容字段

@app.post("/next_item/", 
            summary="切换到下一项",
            description="处理前端下一项按钮点击，增加当前索引并返回对应数据。")
async def next_item(request: NextItemRequest) -> Dict[str, Any]:
    """
    处理前端下一项按钮点击，增加当前索引并返回对应数据。
    """
    try:
        global global_current_index, global_memory_dict, global_desc_polished_all_project, global_polish_suggestions
        polish_suggestions = global_polish_suggestions

        global_memory_dict[global_current_index]['polished_project'] = request.center_content

        # 将字符串索引转换为整数，增加1，再转换回字符串
        current_int = int(global_current_index)
        new_int = min(current_int + 1, len(global_desc_polished_all_project))
        
        global_current_index = str(new_int)

        current_index = global_current_index
        memory_dict = global_memory_dict
        desc_polished_all_project = global_desc_polished_all_project
        header = functions.extract_header(polish_suggestions, current_index)
        results = {
            "success": True,
            "current_index": current_index,
            'header':header,
            "memory_dict": memory_dict,
            "desc_polished_all_project": desc_polished_all_project,
            "message": "处理成功"
        }
        
        return results
        
    except Exception as e:
        print(f"切换下一项时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切换下一项时发生错误: {str(e)}")

@app.post("/previous_item/", 
            summary="切换到上一项",
            description="处理前端上一项按钮点击，减少当前索引并返回对应数据。")
async def previous_item(request: PreviousItemRequest) -> Dict[str, Any]:
    """
    处理前端上一项按钮点击，减少当前索引并返回对应数据。
    """
    try:
        global global_current_index, global_memory_dict, global_desc_polished_all_project, global_polish_suggestions
        polish_suggestions = global_polish_suggestions

        global_memory_dict[global_current_index]['polished_project'] = request.center_content
    
        # 将字符串索引转换为整数，减少1，再转换回字符串
        current_int = int(global_current_index)
        new_int = max(current_int - 1, 1)  # 确保索引不小于1
        global_current_index = str(new_int)

        current_index = global_current_index
        memory_dict = global_memory_dict
        desc_polished_all_project = global_desc_polished_all_project
        header = functions.extract_header(polish_suggestions, current_index)
        results = {
            "success": True,
            "current_index": current_index,
            'header':header,
            "memory_dict": memory_dict,
            "desc_polished_all_project": desc_polished_all_project,
            "message": "处理成功"
        }
        
        return results
        
    except Exception as e:
        print(f"切换上一项时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切换上一项时发生错误: {str(e)}")

# 在现有的请求模型后添加新的请求模型
class PolishProjectRequest(BaseModel):
    user_input: str
    current_index: str
    center_content: str = "" 

# 在现有API端点后添加新的端点
@app.post("/polish_project/", 
            summary="优化指定项目",
            description="接收用户输入，调用polishing_project函数优化指定项目。")
async def polish_project_endpoint(request: PolishProjectRequest) -> Dict[str, Any]:
    """
    处理第五张卡片及后续卡片的发送按钮功能。
    调用polishing_project函数，保存记忆到memory_dict，返回给前端。
    """
    try:
        global global_current_index, global_memory_dict, global_desc_polished_all_project, global_polish_suggestions
        global global_job_title, global_resume_text, global_initial_results, global_all_rate_results, global_personalization
        
        # 获取全局变量
        current_index = request.current_index
        user_input = request.user_input
        memory_dict = global_memory_dict
        job_title = global_job_title
        resume_text = global_resume_text
        initial_results = global_initial_results
        all_rate_results = global_all_rate_results
        personalization = global_personalization
        polish_suggestions = global_polish_suggestions
        
        # 更新当前项目的 polished_project
        global_memory_dict[current_index]['polished_project'] = request.center_content
        
        # 调用polishing_project函数
        desc_polished_project = await asyncio.to_thread(
            functions.polishing_project,
            current_index,
            user_input,
            memory_dict,
            job_title,
            resume_text,
            initial_results,
            all_rate_results,
            personalization
        )
        
        # 提取优化后的项目内容
        polished_project = functions.extract_content_between_dashes(desc_polished_project)
        
        if polished_project == '' or polished_project == None:
            updated_memory_dict = functions.add_memory(
                current_index,
                memory_dict,
                user_message=user_input,
                AI_message=desc_polished_project,
                polished_project=request.center_content
            )
        else:
            # 保存记忆到memory_dict
            updated_memory_dict = functions.add_memory(
                current_index,
                memory_dict,
                user_message=user_input,
                AI_message=desc_polished_project,
                polished_project=polished_project
            )
        
        # 更新全局变量
        global_memory_dict = updated_memory_dict
        global_desc_polished_all_project[current_index] = desc_polished_project
        
        # 获取header
        header = functions.extract_header(polish_suggestions, current_index)
        
        results = {
            "success": True,
            "current_index": current_index,
            "header": header,
            "memory_dict": updated_memory_dict,
            "desc_polished_all_project": global_desc_polished_all_project,
            "message": "项目优化成功"
        }
        
        return results
        
    except Exception as e:
        print(f"优化项目时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"优化项目时发生错误: {str(e)}")
        

@app.post("/save_current_item/", 
            summary="保存当前项",
            description="保存当前项的数据，不进行页面切换。")
async def save_current_item(request: NextItemRequest) -> Dict[str, Any]:
    """
    保存当前项的数据，不进行页面切换。
    """
    try:
        global global_current_index, global_memory_dict
        
        # 保存当前项的数据到memory_dict
        global_memory_dict[global_current_index]['polished_project'] = request.center_content
        
        results = {
            "success": True,
            "message": "当前项数据保存成功"
        }
        
        return results
        
    except Exception as e:
        print(f"保存当前项时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存当前项时发生错误: {str(e)}")



class ConfirmFinishRequest(BaseModel):
    action: str = "confirm_finish"
    center_content: str = ""

@app.post("/confirm_finish/", 
            summary="确认完成",
            description="处理前端确认完成按钮点击，保存数据并返回最终结果。")
async def confirm_finish(request: ConfirmFinishRequest) -> Dict[str, Any]:
    """
    处理前端确认完成按钮点击，保存数据并返回最终结果。
    """
    try:
        global global_current_index, global_memory_dict, global_job_title, global_resume_text, global_initial_results, global_all_rate_results, global_polish_suggestions
        job_title = global_job_title
        resume_text = global_resume_text
        initial_results = global_initial_results
        all_rate_results = global_all_rate_results
        memory_dict = global_memory_dict
        polish_suggestions = global_polish_suggestions

        # 保存当前中间容器的内容到memory_dict
        if global_current_index in memory_dict:
            memory_dict[global_current_index]['polished_project'] = request.center_content

        # 提取所有有效项目的headers
        headers = functions.extract_headers_for_valid_projects(polish_suggestions, memory_dict)

        polished_projects = functions.integrate_polished_projects(memory_dict)
        AI_comment = await asyncio.to_thread(
            functions.AI_comment,
            job_title,resume_text,polished_projects,initial_results,all_rate_results
        )

        # 返回结果，包含headers
        results = {
            "success": True,
            "integrated_results": polished_projects,
            "AI_comments": AI_comment,
            "headers": headers,  # 新增：返回headers
            "message": "确认完成处理成功"
        }
        
        return results
        
    except Exception as e:
        print(f"确认完成时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"确认完成时发生错误: {str(e)}")


# 在所有API路由定义完成后，添加静态文件服务（在第555行之前）
if os.path.exists("my-resume-analyzer-frontend/dist"):
    # 挂载assets目录
    app.mount("/assets", StaticFiles(directory="my-resume-analyzer-frontend/dist/assets"), name="assets")
    
    # 为前端SPA路由提供支持
    @app.get("/edit-resume")
    @app.get("/details")
    @app.get("/results")
    async def serve_spa_routes():
        return FileResponse("my-resume-analyzer-frontend/dist/index.html")
    
    # 根路径返回index.html
    @app.get("/")
    async def serve_index():
        return FileResponse("my-resume-analyzer-frontend/dist/index.html")

if __name__ == "__main__":
    import uvicorn
    # 支持环境变量端口配置
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


    
