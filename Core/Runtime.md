# Runtime

`Runtime[R]` 은 `R` 환경에서 작업을 실행할 수 있다.

이펙트를 실행하기 위해 이펙트를 실행할 수 있는 `Runtime` 이 필요하다.</br>
`Runtime` 들은 이펙트에 필요한 환경과 쓰레드 풀을 묶는다.

## What is a Runtime System?

`ZIO` 생성자와 컴비네이터로부터 `ZIO` 이펙트를 만들어 `ZIO` 프로그램을 작성할 수 있다. `ZIO` 이펙트는 그저 동시 프로그램을 묘사하는 데이터 구조일 뿐이며, 이걸로 청사진을 그린다.</br>
그래서 결국에는 `ZIO` 이펙트의 동작을 묘사하는 서로 다른 여러 데이터 구조들이 합쳐진 하나의 트리 데이터 구조가 만들어진다. 이렇게 만들어진 데이터 구조는 동시 프로그램의 설명일 뿐, **아무 동작도 하지 않는다.**

`ZIO` 와 비슷한 함수형 이펙트 시스템으로 작업할 때 가장 중요한 점은, 우린 그저어플리케이션에 대한 하나의 워크플로우나 청사진을 작성한다는 것이다. (코드를 작성하거나, 문자열을 콘솔에 찍는다거나, 파일을 읽거나, db 에 쿼리를 날린다거나 등의)</br>
데이터 구조를 만들 뿐이다.

`ZIO` 는 어떻게 이 워크플로우들을 실행할까?</br>
`ZIO Runtime System` 이 이를 가능케 한다. `unsaferun` 함수를 실행할 때마다 `ZIO` 이펙트로 만들어진 모든 명령들이 단계적으로 실행된다.

간단하게 생각하면, `Runtime System` 은 `ZIO` 이펙트(`ZIO[R, E, A]`)와 환경(`R`)으로 이루어진 하나의 블랙 박스라고 볼 수 있다.</br>
`Runtim System` 은 이를 실행하고 결과값을 `Either[E, A]` 값으로 반환한다.

![ZIO Runtime System](../img/zio-runtime-system.svg)

## Responsibilities of the Runtime System

`Runtime System` 들은 다음과 같은 많은 책임이 있다:

1. **청사진의 모든 스텝 실행** — 하나의 while 루프에서 모든 스텝이 실행되고 완료되어야 한다.

2. **예기치 못한 오류 처리** — 예상된 오류 뿐만 아니라, 예상치 못한 오류도 처리해야 한다.

3. **동시성 fiber 생성** — 이펙트 시스템의 동시성에 대한 책임을 가진다. 이펙트에 `fork` 를 호출할 때마다 새로운 fiber 를 만들어야 한다.

4. **다른 fiber 와의 협력** — CPU 자원을 독점할 수 있는 다른 fiber 와 협력하여 fiber 간의 CPU 자원을 적절히 분배해야 한다.

5. **실행 및 스택 추적 캡쳐** — 사용자 영역 코드의 실행 상황을 추적하여 더 나은 추적 결과가 캡쳐되어야 한다. (더 나은 로그 제공)

6. **보장된 종료자** — clean-up 로직이 수행되었을 때 자원이 정상적으로 종료되길 보장해야 하므로 어떤 상황에서든 종료자는 정상적으로 실행됨이 보장되어야 한다. 이것은 ZIO 의 Scope 과 다른 자원 안정적인 구조를 강화하는 기능이다.

7. **비동기 콜백 제어** — 비동기 콜백을 처리하는 복잡한 로직을 대신 제어함으로써 비동기 코드를 다루지 않아도 된다. ZIO 를 사용할 때, 모든 것은 실행된 비동기(async out of the box)이다.

## Running a ZIO Effect

`ZIO Effect` 를 실행하는 일반적인 두 가지 방법이 있다.</br>
대부분의 경우 [`ZIOAppDefault`](ZIOAPP.md) 트레이트를 사용한다. </br>
하지만 `unsafeRun` 을 사용해 `ZIO Effect` 를 런타임 시스템에 직접 주입하는 방법을 커스텀할 수 있다:
```scala mdoc:compile-only
import zio._

object RunZIOEffectUsingUnsafeRun extends scala.App {
  val myAppLogic = for {
    _ <- Console.printLine("Hello! What is your name?")
    n <- Console.readLine
    _ <- Console.printLine("Hello, " + n + ", good to meet you!")
  } yield ()

  Unsafe.unsafe { implicit unsafe =>
      zio.Runtime.default.unsafe.run(
        myAppLogic
      ).getOrThrowFiberFailure()
  }
}
```

위 메소드는 자주 사용되지 않지만, non-effectful 한 레거시 코드를 ZIO 로 통합하는 경우 등에 사용된다. 이는 방대한 레거시 코드를 ZIO 로 점진적으로 이동하는 데에 도움을 준다.</br>
레거시 코드의 중간에 있는 컴포넌트 코드를 ZIO 로 재작성한다고 가정해 보자. 해당 컴포넌트를 ZIO Effect 로 재작성하고 `unsafeRun` 함수를 사용해 기존 코드에서 동작하게 할 수 있다.

## Default Runtime

`ZIO` 에는 주로 사용되게 만들어진 `Runtime.default` 라는 기본 런타임이 있다. 다음과 같이 구현되어 있다:
```scala
object Runtime {
  val default: Runtime[Any] =
    Runtime(ZEnvironment.empty, FiberRefs.empty, RuntimeFlags.default)
}
```

이 기본 런타임은 `ZIO` 작업을 부트스트랩 실행하는 데에 최소 기능이 포함되어 있다.</br>
다음과 같이 쉽게 기본 `Runtime` 으로 Effect 를 실행할 수 있다:
```scala mdoc:compile-only
object MainApp extends scala.App {
  val myAppLogic = ZIO.succeed(???)

  val runtime = Runtime.default

  Unsafe.unsafe { implicit unsafe =>
    runtime.unsafe.run(myAppLogic).getOrThrowFiberFailure()
  }
}
```

## Top-level And Locally Scoped Runtimes

`ZIO` 엔 두 종류의 런타임이 있다:

- **최상위 런타임** 은 최초에 전체 `ZIO` 어플리케이션을 구동한다. 단 하나의 최상위 런타임만 존재할 수 있으며 사용 예는 다음과 같다:
  - 혼합 어플리케이션에서 최상위 런타임 만들기 : `ZIO` 를 지원하지 않는 HTTP 라이브러리를 사용한다면 각각의 라우트에서 `Runtime.unsafe.run` 을 사용할 수 있다.
  - 어플리케이션 최초에 커스텀 모니터링 혹은 supercvisor 를 설치하는 경우

- **지역적 런타임** 은 `ZIO` 어플리케이션의 실행 중에 사용된다. 이는 특정 구간의 로컬 코드이다.</br>
`ZIO` 어플리케이션의 중간에서 런타임 설정을 바꾸려는 걸 생각해 보자. 이는 지역적 런타임이다. 예를 들면:
  - 특정 런타임에 effectful 혹은 부수 효과가 있는 어플리케이션을 import 하고 싶은 경우
  - 성능 이슈가 있는 구간에 로깅을 비활성화 하고 싶은 경우
  - 부분 코드 실행을 위한 커스텀 실행자를 사용하고 싶은 경우

`ZLayer` 는 런타임을 커스텀하거나 설정하는 일관된 방법을 제공한다. 레이어를 사용해 런타임을 커스텀함으로써, `ZIO` 워크플로를 사용할 수 있다. 따라서 설정 워크플로는 순수거나, effectful 하거나 resourceful 할 수 있다.</br>
파일 혹은 데이터베이스의 설정 정보를 기반으로 런타임을 커스터마이징한다고 가정해 보자.

대부분의 경우 [`bootstrap` layer](#configuring-runtime-using-bootstrap-layer) 혹은 [providing a custom configuration](#configuring-runtime-by-providing-configuration-layers) 을 사용해 어플리케이션 런타임을 커스텀하는 것으로 충분하다.</br>
이 방법이 맞지 않는다면 [top-level runtime configurations](#top-level-runtime-configuration) 을 사용해 볼 수 있다.

각각의 솔루션을 자세히 알아보자.

## Locally Scoped Runtime Configuration

`ZIO` 에서 모든 런타임 설정들은 부모 워크플로로부터 물려받는다. 따라서 런타임 설정에 접근하거나, 워크플로 내부의 런타임을 얻을 때마다 런타임의 부모 워크플로의에 접근하게 된다.</br>
특정 지역의 코드에 새로운 설정을 제공함으로써 부모 워크플로의 런타임 설정을 덮어쓸 수 있다. 이를 지역 범위의 런타임 설정이라고 한다.</br>
해당 지역의 실행이 끝나면 런타임 설정은 다시 기존 값으로 돌아간다.

`ZIO#provideXYZ` 는 특정 지역 코드에 새로운 런타임 설정을 제공하기 위해 사용되는 연산자이다: (++)

### Configuring Runtime by Providing Configuration Layers

`ZIO#provideXYZ` 로 `ZIO` 워크플로에 런타임 설정을 쉽게 바꿀 수 있다: (for 문에 provide)
```scala mdoc:compile-only
import zio._

object MainApp extends ZIOAppDefault {
  val addSimpleLogger: ZLayer[Any, Nothing, Unit] =
    Runtime.addLogger((_, _, _, message: () => Any, _, _, _, _) => println(message()))

  def run = {
    for {
      _ <- ZIO.log("Application started!")
      _ <- ZIO.log("Application is about to exit!")
    } yield ()
  }.provide(Runtime.removeDefaultLoggers ++ addSimpleLogger)
}
```

결과:

```scala
Application started!
Application is about to exit!
```

`ZIO` 어플리케이션의 특정 지역에 런타임 설정을 추가하기 위해 설정 레이어를 추가할 수도 있다: (for 문 내부 provide)
```scala mdoc:compile-only
import zio._

object MainApp extends ZIOAppDefault {
  val addSimpleLogger: ZLayer[Any, Nothing, Unit] =
    Runtime.addLogger((_, _, _, message: () => Any, _, _, _, _) => println(message()))

  def run =
    for {
      _ <- ZIO.log("Application started!")
      _ <- {
        for {
          _ <- ZIO.log("I'm not going to be logged!")
          _ <- ZIO.log("I will be logged by the simple logger.").provide(addSimpleLogger)
          _ <- ZIO.log("Reset back to the previous configuration, so I won't be logged.")
        } yield ()
      }.provide(Runtime.removeDefaultLoggers)
      _ <- ZIO.log("Application is about to exit!")
    } yield ()
}
```

결과:

```scala
timestamp=2022-08-31T14:28:34.711461Z level=INFO thread=#zio-fiber-6 message="Application started!" location=<empty>.MainApp.run file=ZIOApp.scala line=9
I will be logged by the simple logger.
timestamp=2022-08-31T14:28:34.832035Z level=INFO thread=#zio-fiber-6 message="Application is about to exit!" location=<empty>.MainApp.run file=ZIOApp.scala line=17
```

### Configuring Runtime Using `bootstrap` Layer

`bootstrap` 레이어는 어플리케이션 구동에 필요한 서비스들을 획득/해제하는데에 사용되는 특별한 레이어다.</br>
하지만 이 또한 런타임 커스터마이징에 적용할 수 있다. `ZIOAPP` 트레이트의 `bootstrap` 레이어를 오버라이드하면 된다.

최상위 런타임의 초기화 이후 이걸 적용하면 `run` 메소드를 통해 `ZIO` 어플리케이션에 `bootstrap` 레이어를 적용할 수 있다.
```scala mdoc:compile-only
import zio._

object MainApp extends ZIOAppDefault {
  val addSimpleLogger: ZLayer[Any, Nothing, Unit] =
    Runtime.addLogger((_, _, _, message: () => Any, _, _, _, _) => println(message()))

  override val bootstrap: ZLayer[Any, Nothing, Unit] =
    Runtime.removeDefaultLoggers ++ addSimpleLogger

  def run =
    for {
      _ <- ZIO.log("Application started!")
      _ <- ZIO.log("Application is about to exit!")
    } yield ()
}
```

결과:
```scala
Application started!
Application is about to exit!
```

이 메소드를 사용하면 전체 `ZIO` 어플리케이션에 적용되지만, 지역적 런타임 설정으로 분류된다. `bootstrap` 레이어는 최상위 런타임이 초기화된 이후 평가/적용되기 때문이다.</br>
따라서 이는 `run` 메소드를 통해서만 적용될 수 있다.

이에 대한 더 자세한 설명은 다음 예시를 보라:
```scala mdoc:compile-only
import zio._

object MainApp extends ZIOAppDefault {
  val addSimpleLogger: ZLayer[Any, Nothing, Unit] =
    Runtime.addLogger((_, _, _, message: () => Any, _, _, _, _) => println(message()))
  
  val effectfulConfiguration: ZLayer[Any, Nothing, Unit] =
    ZLayer.fromZIO(ZIO.log("Started effectful workflow to customize runtime configuration"))

  override val bootstrap: ZLayer[Any, Nothing, Unit] =
    Runtime.removeDefaultLoggers ++ addSimpleLogger ++ effectfulConfiguration

  def run =
    for {
      _ <- ZIO.log("Application started!")
      _ <- ZIO.log("Application is about to exit!")
    } yield ()
}
```

어떤 결과가 예상되나?</br>
`Runtime.removeDefaultLoggers` 로 인해 런타임에서 default logger 가 제거되어 simple logger 로만 로그 메시지가 찍히기를 기대했을 것이다.</br>
하지만 그렇지 않다. 최상위 런타임이 초기화된 후 `effectfulConfiguration` 가 평가된다. 따라서 `effectfulConfiguration` 레이어의 초기화와 관련된 로그 메시지는 default logger 로부터 출력받을 수 있다. (지우기 전이므로)
```scala
timestamp=2022-09-01T08:07:47.870219Z level=INFO thread=#zio-fiber-6 message="Started effectful workflow to customize runtime configuration" location=<empty>.MainApp.effectfulConfiguration file=ZIOApp.scala line=8
Application started!
Application is about to exit!
```

## Top-level Runtime Configuration

When we write a ZIO application using the `ZIOAppDefault` trait, a default top-level runtime is created and used to run the application automatically under the hood. Further, we can customize the rest of the ZIO application by providing locally scoped configuration layers using [`provideXYZ` operations](#configuring-runtime-by-providing-configuration-layers) or [`bootstrap` layer](#configuring-runtime-using-bootstrap-layer).

This is usually sufficient for lots of ZIO applications, but it is not always the case. There are cases where we want to customize the runtime of the entire ZIO application from the top level.

In such cases, we need to create a top-level runtime by unsafely running the configuration layer to convert that configuration to the `Runtime` by using the `Runtime.unsafe.fromLayer` operator:

```scala mdoc:invisible
import zio._
val layer = ZLayer.empty
```

```scala mdoc:compile-only
val runtime: Runtime[Any] =
  Unsafe.unsafe { implicit unsafe =>
    Runtime.unsafe.fromLayer(layer)
  }
```

Let's try a fully working example:

```scala mdoc:compile-only
import zio._

object MainApp extends ZIOAppDefault {

  // In a real-world application we might need to implement a `sl4jlogger` layer
  val addSimpleLogger: ZLayer[Any, Nothing, Unit] =
    Runtime.addLogger((_, _, _, message: () => Any, _, _, _, _) => println(message()))

  val layer: ZLayer[Any, Nothing, Unit] =
    Runtime.removeDefaultLoggers ++ addSimpleLogger

  override val runtime: Runtime[Any] =
    Unsafe.unsafe { implicit unsafe =>
      Runtime.unsafe.fromLayer(layer)
    }

  def run = ZIO.log("Application started!")
}
```

:::caution
Keep in mind that only the "bootstrap" layer of applications will be combined when we compose two ZIO applications. Therefore, when we compose two ZIO programs, top-level runtime configurations won't be integrated.
:::

Another use-case of top-level runtimes is when we want to integrate our ZIO application inside a legacy application:

```scala mdoc:compile-only
import zio._

object MainApp {
  val sl4jlogger: ZLogger[String, Any] = ???

  def legacyApplication(input: Int): Unit = ???

  val zioWorkflow: ZIO[Any, Nothing, Int] = ???

  val runtime: Runtime[Unit] =
    Unsafe.unsafe { implicit unsafe =>
      Runtime.unsafe
        .fromLayer(
          Runtime.removeDefaultLoggers ++ Runtime.addLogger(sl4jlogger)
        )
    }

  def zioApplication(): Int =
    Unsafe.unsafe { implicit unsafe =>
      runtime.unsafe
        .run(zioWorkflow)
        .getOrThrowFiberFailure()
    }

  def main(args: Array[String]): Unit = {
    val result = zioApplication()
    legacyApplication(result)
  }

}
```

## Providing Environment to Runtime System

The custom runtime can be used to run many different effects that all require the same environment, so we don't have to call `ZIO#provide` on all of them before we run them.

For example, assume we want to create a `Runtime` for services that are for testing purposes, and they don't interact with real external APIs. So we can create a runtime, especially for testing.

Let's say we have defined two `LoggingService` and `EmailService` services:

```scala mdoc:silent:nest
trait LoggingService {
  def log(line: String): UIO[Unit]
}

object LoggingService {
  def log(line: String): URIO[LoggingService, Unit] =
    ZIO.serviceWith[LoggingService](_.log(line))
}

trait EmailService {
  def send(user: String, content: String): Task[Unit]
}

object EmailService {
  def send(user: String, content: String): ZIO[EmailService, Throwable, Unit] =
    ZIO.serviceWith[EmailService](_.send(user, content))
}
```

We are going to implement a live version of `LoggingService` and also a fake version of `EmailService` for testing:

```scala mdoc:silent:nest
case class LoggingServiceLive() extends LoggingService {
  override def log(line: String): UIO[Unit] =
    ZIO.succeed(print(line))
}

case class EmailServiceFake() extends EmailService {
  override def send(user: String, content: String): Task[Unit] =
    ZIO.attempt(println(s"sending email to $user"))
}
```

Let's create a custom runtime that contains these two service implementations in its environment:

```scala mdoc:silent:nest
val testableRuntime = Runtime(
  ZEnvironment[LoggingService, EmailService](LoggingServiceLive(), EmailServiceFake()),
  FiberRefs.empty,
  RuntimeFlags.default
)
```

Also, we can replace the environment of the default runtime with our own custom environment, which allows us to add new services to the ZIO environment:

```scala mdoc:silent:nest
val testableRuntime: Runtime[LoggingService with EmailService] =
  Runtime.default.withEnvironment {
    ZEnvironment[LoggingService, EmailService](LoggingServiceLive(), EmailServiceFake())
  }
```

Now we can run our effects using this custom `Runtime`:

```scala mdoc:silent:nest
Unsafe.unsafe { implicit unsafe =>
    testableRuntime.unsafe.run(
      for {
        _ <- LoggingService.log("sending newsletter")
        _ <- EmailService.send("David", "Hi! Here is today's newsletter.")
      } yield ()
    ).getOrThrowFiberFailure()
}
```
