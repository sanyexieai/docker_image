"""
搜索引擎封装模块
支持 DuckDuckGo 和 Sogou 两种搜索方式
"""

import time
import logging
import json
import os
from typing import List, Dict, Any
from duckduckgo_search import DDGS
from app.utils.googlenews_utils import GoogleNewsSearch
from datetime import datetime, timedelta

# 尝试导入搜狗搜索，如果失败则只支持DDG
try:
    from sogou_search import sogou_search
    SOGOU_AVAILABLE = True
except ImportError:
    SOGOU_AVAILABLE = False

class SearchEngine:
    """搜索引擎封装类，支持多引擎合并去重"""

    def __init__(self, engine=None, cache_dir="search_cache", cache_expire_days=3):
        """
        初始化搜索引擎

        Args:
            engine: 搜索引擎类型，支持 "ddg" (DuckDuckGo)、"sogou" (搜狗)、"google" (Google News) 或它们的列表。若为空则默认全部。
            cache_dir: 缓存目录
            cache_expire_days: 缓存过期天数，默认3天
        """
        # 初始化日志记录器
        self.setup_logging()
        
        # 缓存配置
        self.cache_dir = cache_dir
        self.cache_expire_days = cache_expire_days
        os.makedirs(self.cache_dir, exist_ok=True)
        
        all_supported = ["ddg", "sogou", "google"]
        # 处理 engine 参数为空的情况
        if not engine or (isinstance(engine, (list, tuple)) and not engine):
            self.engines = all_supported.copy()
        elif isinstance(engine, str):
            if engine.strip() == "":
                self.engines = all_supported.copy()
            else:
                self.engines = [engine.lower()]
        else:
            self.engines = [e.lower() for e in engine]
        self.delay = 1.0  # 默认搜索延迟

        # 检查支持性
        valid_engines = []
        for e in self.engines:
            if e not in all_supported:
                self.logger.warning(f"警告: 不支持的搜索引擎: {e}. 跳过。")
                continue
            if e == "sogou" and not SOGOU_AVAILABLE:
                self.logger.warning("警告: 搜狗搜索 (k_sogou_search) 未安装, 跳过搜狗。")
                continue
            valid_engines.append(e)
        if not valid_engines:
            error_msg = "未指定有效的搜索引擎。"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.engines = valid_engines
        self.logger.info(f"🔍 搜索引擎初始化完成，使用引擎: {self.engines}")
        self.logger.info(f"📁 缓存目录: {self.cache_dir}")
        self.logger.info(f"⏰ 缓存过期时间: {self.cache_expire_days} 天")
    
    def setup_logging(self):
        """配置日志记录"""
        # 创建logs目录
        import os
        os.makedirs("logs", exist_ok=True)
        
        # 生成日志文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"logs/search_engine_{timestamp}.log"
        
        # 配置日志记录器
        self.logger = logging.getLogger('SearchEngine')
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"📝 搜索引擎日志记录已启动，日志文件: {log_filename}")
    
    def _get_cache_key(self, keywords: str, max_results: int, start_date=None, end_date=None) -> str:
        """生成缓存键"""
        import hashlib
        cache_data = {
            'keywords': keywords,
            'max_results': max_results,
            'start_date': start_date,
            'end_date': end_date,
            'engines': self.engines
        }
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Dict[str, Any]:
        """从缓存加载数据"""
        cache_file = self._get_cache_file_path(cache_key)
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查缓存是否过期
            cache_time = datetime.fromisoformat(cache_data['cache_time'])
            expire_time = cache_time + timedelta(days=self.cache_expire_days)
            
            if datetime.now() > expire_time:
                self.logger.info(f"📅 缓存已过期，过期时间: {expire_time}")
                return None
            
            self.logger.info(f"📁 从缓存加载数据，缓存时间: {cache_time}")
            return cache_data
        except Exception as e:
            self.logger.error(f"❌ 读取缓存失败: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, search_results: List[Dict[str, Any]], 
                      keywords: str, max_results: int, start_date=None, end_date=None):
        """保存数据到缓存"""
        cache_file = self._get_cache_file_path(cache_key)
        try:
            cache_data = {
                'cache_time': datetime.now().isoformat(),
                'keywords': keywords,
                'max_results': max_results,
                'start_date': start_date,
                'end_date': end_date,
                'engines': self.engines,
                'results': search_results
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 搜索结果已缓存到: {cache_file}")
        except Exception as e:
            self.logger.error(f"❌ 保存缓存失败: {e}")

    def search(self, keywords: str, max_results: int = 10, start_date=None, end_date=None, 
               force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        统一搜索接口，支持多引擎合并去重和本地缓存

        Args:
            keywords: 搜索关键词
            max_results: 每个引擎最大结果数
            start_date, end_date: 仅对 google 有效，格式 yyyy-mm-dd
            force_refresh: 是否强制刷新，True时忽略缓存直接搜索

        Returns:
            搜索结果列表，每个结果包含 title, url, description, engine 字段
        """
        # 生成缓存键
        cache_key = self._get_cache_key(keywords, max_results, start_date, end_date)
        
        # 检查是否需要强制刷新
        if force_refresh:
            self.logger.info(f"🔄 强制刷新模式，忽略缓存直接搜索")
        else:
            # 尝试从缓存加载
            cache_data = self._load_from_cache(cache_key)
            if cache_data:
                self.logger.info(f"📁 使用缓存数据，关键词: '{keywords}'")
                return cache_data['results']
            else:
                self.logger.info(f"📁 缓存不存在或已过期，进行网络搜索")
        
        self.logger.info(f"🔍 开始搜索，使用引擎: {','.join([e.upper() for e in self.engines])}")
        self.logger.info(f"📝 搜索关键词: '{keywords}'")
        self.logger.info(f"📊 最大结果数: {max_results}")
        if start_date or end_date:
            self.logger.info(f"📅 时间范围: {start_date} 至 {end_date}")
        
        all_results = {}
        total_results = 0
        
        for engine in self.engines:
            try:
                self.logger.info(f"🚀 开始 {engine.upper()} 搜索...")
                start_time = time.time()
                
                if engine == "ddg":
                    results = self._search_ddg(keywords, max_results)
                elif engine == "sogou":
                    results = self._search_sogou(keywords, max_results)
                elif engine == "google":
                    results = self._search_google(keywords, max_results, start_date, end_date)
                else:
                    results = []
                
                search_time = time.time() - start_time
                self.logger.info(f"✅ {engine.upper()} 搜索完成，耗时: {search_time:.2f}秒，获得 {len(results)} 个结果")
                
                # 处理搜索结果
                new_results = 0
                duplicate_results = 0
                for r in results:
                    url = r.get('url')
                    if not url:
                        continue
                    if url in all_results:
                        # 已有，合并 engine 字段
                        if engine not in all_results[url]['engine']:
                            all_results[url]['engine'].append(engine)
                        duplicate_results += 1
                    else:
                        # 新结果
                        r['engine'] = [engine]
                        all_results[url] = r
                        new_results += 1
                
                self.logger.info(f"📊 {engine.upper()} 结果统计 - 新增: {new_results}, 重复: {duplicate_results}")
                total_results += len(results)
                
                time.sleep(self.delay)
                
            except Exception as e:
                self.logger.error(f"❌ {engine.upper()} 搜索失败: {e}")
                self.logger.error(f"🔍 失败详情 - 关键词: {keywords}, 最大结果数: {max_results}")
        
        # 最终统计
        final_count = len(all_results)
        self.logger.info(f"🎯 搜索完成！总获得 {total_results} 个原始结果，去重后 {final_count} 个唯一结果")
        self.logger.info(f"📋 去重率: {((total_results - final_count) / total_results * 100):.1f}%" if total_results > 0 else "0%")
        
        # 保存到缓存
        final_results = list(all_results.values())
        self._save_to_cache(cache_key, final_results, keywords, max_results, start_date, end_date)
        
        # 返回合并后的结果
        return final_results

    def _search_ddg(self, keywords: str, max_results: int) -> List[Dict[str, Any]]:
        """DuckDuckGo 搜索"""
        self.logger.info(f"🦆 使用 DuckDuckGo 搜索: {keywords}")
        try:
            results = DDGS().text(
                keywords=keywords,
                region="cn-zh",
                max_results=max_results
            )
            # 标准化结果格式
            formatted_results = [
                {
                    'title': r.get('title', '无标题'),
                    'url': r.get('href', '无链接'),
                    'description': r.get('body', '无摘要')
                }
                for r in results
            ]
            self.logger.info(f"✅ DuckDuckGo 搜索成功，获得 {len(formatted_results)} 个结果")
            return formatted_results
        except Exception as e:
            self.logger.error(f"❌ DuckDuckGo 搜索异常: {e}")
            return []

    def _search_sogou(self, keywords: str, max_results: int) -> List[Dict[str, Any]]:
        """搜狗搜索"""
        self.logger.info(f"🔍 使用搜狗搜索: {keywords}")
        if not SOGOU_AVAILABLE:
            self.logger.warning("⚠️ 搜狗搜索模块未安装，跳过搜狗搜索")
            return []
        try:
            # 假设 sogou_search 返回的结果已包含 title, url, description 字段
            results = sogou_search(keywords, num_results=max_results)
            self.logger.info(f"✅ 搜狗搜索成功，获得 {len(results)} 个结果")
            return results
        except Exception as e:
            self.logger.error(f"❌ 搜狗搜索异常: {e}")
            return []

    def _search_google(self, keywords: str, max_results: int, start_date=None, end_date=None) -> List[Dict[str, Any]]:
        """Google News 搜索"""
        self.logger.info(f"📰 使用 Google News 搜索: {keywords}")
        # 如果没有传 start_date/end_date，则不加时间限制
        if not start_date:
            start_date = None
        if not end_date:
            end_date = None
        
        try:
            results = GoogleNewsSearch.search(keywords, max_results, start_date, end_date)
            self.logger.info(f"✅ Google News 搜索成功，获得 {len(results)} 个结果")
            return results
        except Exception as e:
            self.logger.error(f"❌ Google News 搜索异常: {e}")
            return []
    
    def clear_cache(self, days_old: int = None):
        """
        清理缓存
        
        Args:
            days_old: 清理多少天前的缓存，None表示清理所有缓存
        """
        try:
            cleared_count = 0
            current_time = datetime.now()
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if days_old is None or (current_time - file_time).days > days_old:
                        os.remove(file_path)
                        cleared_count += 1
            
            self.logger.info(f"🧹 缓存清理完成，删除了 {cleared_count} 个缓存文件")
        except Exception as e:
            self.logger.error(f"❌ 缓存清理失败: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_size = 0
            cache_details = []
            
            for filename in cache_files:
                file_path = os.path.join(self.cache_dir, filename)
                file_size = os.path.getsize(file_path)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                total_size += file_size
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    keywords = cache_data.get('keywords', '未知')
                    results_count = len(cache_data.get('results', []))
                except:
                    keywords = '读取失败'
                    results_count = 0
                
                cache_details.append({
                    'filename': filename,
                    'keywords': keywords,
                    'size': file_size,
                    'modified': file_time,
                    'results_count': results_count
                })
            
            return {
                'total_files': len(cache_files),
                'total_size': total_size,
                'cache_dir': self.cache_dir,
                'expire_days': self.cache_expire_days,
                'details': cache_details
            }
        except Exception as e:
            self.logger.error(f"❌ 获取缓存信息失败: {e}")
            return {}

if __name__ == "__main__":
    # 测试代码
    print("测试搜索引擎封装...")

    # 测试多引擎合并
    print("\n=== 测试多引擎合并 ===")
    # engines = ["ddg", "google"]
    # if SOGOU_AVAILABLE:
    #     engines.append("sogou")
    multi_engine = SearchEngine()
    multi_results = multi_engine.search("商汤科技 行业地位 市场份额 业务模式 发展战略", max_results=2, start_date="2024-01-01", end_date="2024-06-01")
    for i, result in enumerate(multi_results, 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   描述: {result['description'][:100]}...")
        print(f"   来源引擎: {result['engine']}")
        if 'date' in result:
            print(f"   日期: {result.get('date', '')}")
        if 'source' in result:
            print(f"   来源: {result.get('source', '')}")
    
    # 测试缓存功能
    print("\n=== 测试缓存功能 ===")
    cache_info = multi_engine.get_cache_info()
    print(f"缓存信息: {cache_info}")
