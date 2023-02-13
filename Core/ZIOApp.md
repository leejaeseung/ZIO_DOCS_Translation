# ZIOApp

`ZIOApp` trait 는 `ZIO` 어플리케이션 레이어 간 공유를 가능하게 하는 시작점이다.</br>
여러 개의 `ZIO` 어플리케이션을 합성할 수도 있게 해준다.

`ZIOAppDefault` 는 `ZIOApp` 의 단순한 버전이다.</br>
보통의 경우 `ZIO` 의 기본 환경인 `ZEnv` 를 사용하는 `ZIOAppDefault` 를 사용한다.

## Running a ZIO effect

`ZIOAppDefault` 의 `run` 함수는 JVM 환경에서 `ZIO` 어플리케이션을 실행하기 위한 시작점을 제공한다:
```scala mdoc:compile-only
import zio._

object MyApp extends ZIOAppDefault {
  def run = for {
    _ <- Console.printLine("Hello! What is your name?")
    n <- Console.readLine
    _ <- Console.printLine("Hello, " + n + ", good to meet you!")
  } yield ()
}
```

## Accessing Command-line Arguments

`ZIO` 에는 `ZIOAppArgs` 라 불리는 커맨드라인 인자가 포함된 서비스가 존재한다.</br>
미리 정의된 `getArgs` 함수로 커맨드라인 인자에 접근할 수 있다:
```scala mdoc:compile-only
import zio._

object HelloApp extends ZIOAppDefault {
  def run = for {
    args <- getArgs
    _ <-
      if (args.isEmpty)
        Console.printLine("Please provide your name as an argument")
      else
        Console.printLine(s"Hello, ${args.head}!")
  } yield ()
}
```

## Customized Runtime

`ZIO` 앱에서, `Runtime` 값을 오버라이드하여 현재 `Runtime` 을 커스텀할 수 있다.</br>
커스텀 `Executor` 세팅:
```scala mdoc:compile-only
import zio._
import zio.Executor
import java.util.concurrent.{LinkedBlockingQueue, ThreadPoolExecutor, TimeUnit}

object CustomizedRuntimeZIOApp extends ZIOAppDefault {
  override val bootstrap = Runtime.setExecutor(
    Executor.fromThreadPoolExecutor(
      new ThreadPoolExecutor(
        5,
        10,
        5000,
        TimeUnit.MILLISECONDS,
        new LinkedBlockingQueue[Runnable]()
      )
    )
  )

  def run = myAppLogic
}
```

더 자세한 `ZIO Runtime` 시스템은 [Runtime](Runtime.md) 에 나와있다.

## Installing Low-level Functionalities

`ZIO` 런타임에 연결하여 `ZIO` 어플리케이션의 로우-레벨 기능들을 설치할 수 있다. (***로깅***, ***Profiling*** 같은 기본 인프라들)

더 자세한 설명은 [Runtime](Runtime.md) 에 나와있다.

## Composing ZIO Applications

`<>` 연산자를 사용해 `ZIO` 어플리케이션을 합성할 수 있다:
```scala mdoc:compile-only
import zio._

object MyApp1 extends ZIOAppDefault {    
  def run = ZIO.succeed(???)
}

object MyApp2 extends ZIOAppDefault {
  override val bootstrap: ZLayer[Any, Any, Any] =
    asyncProfiler ++ slf4j ++ loggly ++ newRelic

  def run = ZIO.succeed(???)
}

object Main extends ZIOApp.Proxy(MyApp1 <> MyApp2)
```

`<>` 연산자는 두 어플리케이션의 레이어를 합치고, 두 어플리케이션을 병렬적으로 실행한다.