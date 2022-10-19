---
tags: React
---
# 从零开始调用web API
## 步骤一 创建工程和本地API

## app.py
- 注册蓝图
```
app.register_blueprint(poetry, url_prefix='/api/v2/poetry')

def run():
    service_config = app.config.get("SERVICE_CONFIG")
    host = service_config.get("host")
    port = service_config.get("port")
    debug = service_config.get("debug")

    app.run(host=host, port=port, debug=False,threaded=False)
```
## view.py
- 装饰器分配路由
```
from GPT_3 import app

@poetry.route('/normal/', methods=["POST"])
def poetry_normal():
    try:
        title = HttpUtil.check_param("title", request, method=1)
        author = HttpUtil.check_param("author", request, method=1)
        emo = HttpUtil.check_param("emo", request, method=1, default='', required=False)
        desc = HttpUtil.check_param("desc", request, method=1, default='', required=False)
        content = joint.join([title, author, emo, desc, ''])
        processLogger.info("写诗请求："+ content)
        data = generator.poem_emo_heading(content)
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


```

## Teller.py
```
写实际调用函数
```

## 步骤二 从API获取数据用useEffect

REACT
## 调用界面
## useEffect()
```
#定义状态
  const status_btn_texts = {
    enable: "开始生成",
    disable: "完善输入",
    loading: "生成中"
  }

#设置钩子
  useEffect(() => {
    if (status !== 'loading') {
      setStatus('disable');
    } else {
      setStatus('loading');
    }
  }, [menuStatus])


btn_status={status}
btn_text={status_btn_texts[status]}
btn_on_click={handleRequest}

#监听
  useEventListener('keydown', (e) => {
    if (e?.keyCode === 13) {
      handleRequest()
    }
  })

# 点击函数
  const handleRequest = useCallback(async (_) => {
    if (status == "enable") {
      const values = await checkFormFieldsByFormInstance(form);
      setStatus("loading")
      let payload = {
        title: values.topic,
        author: values.author,
      };

      modelType.current = values?.model || 'transformer-xl';

      if (menuStatus == "classical") {
        payload.desc = values.desc;model || modelType?.current;
        if (values?.best) {
          payload.emo = values?.emo || '一般';
        }
      } else if (menuStatus == "orz_poem") {
        payload.desc = values.desc;
        payload.heading = values.heading;
        payload.emo = values?.emo || '一般';
      } else if (menuStatus == "song_peams") {
        payload.cipai = values.cipai;
        if (values?.best) {
          payload.emo = values?.emo || '一般';
          payload.old = true;
        }
      }
      dispatch({
        type: "app/wordsFilter",
        payload: {content: payload.topic+payload.author+(payload.desc||"")+(payload.heading||"")}
      }).then(data => {
        if (data) {
          // 判断根据选项调用不同模型的写诗: poem_fast, poem_glm, poem_transform_XL
          dispatch({
            type: menuStatus == "classical" && !values.best && values.model !== 'glm' && "app/write_poem_fast" || "app/write_poem",
            payload: {
              ...payload,
            }
          })
          closeHistory();
        } else {
          message.error("输入信息非法，请修改后重试！！");
          setStatus("enable")
        }
      })

    } else if (status == "loading") {
      message.info("文汇疯狂计算中，请耐心等待...")
    } else {
      await checkFormFieldsByFormInstance(form);
    }
  }, [status, values])


```
这里涉及异步调用阻塞机制，后续完善

## app.js
```
import { postAPI, getAPI, request } from '@/utils'
import { message } from 'antd'

const host = {
  prod: 'http://192.168.249.24:19550',
  test: 'http://192.168.249.24:19550',
  dev: 'http://192.168.249.24:19550',
}

  effects: {
    //Job
    *requestJob({ payload }, { call, put }) {
      yield put({ type: 'handleRequestJobStart', payload })
      try {
        let data = yield call(postAPI, jd_host, '/generate/jd', payload, undefined, { no_token: true })
        if (data && data.status === 0) {
          yield put({ type: 'handleRequestJobSuccess', payload: { status: data.status, a: data.result.output.text } })
        } else {
          message.error('请求失败，请稍后重试')
          yield put({ type: 'handleRequestJobError' })
        }
      } catch (error) {
        message.error('请求失败，请稍后重试')
        yield put({ type: 'handleRequestJobError' })
      }
    },
```