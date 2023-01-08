# ZIO

`ZIO[R, E, A]` 는 워크플로나 잡을 지연 수행할 수 있는 불변 값이다.</br>
워크플로는 환경 타입인 `R` 이 필요하며 에러 타입인 `E` 로 실패할 수도 있고, 성공 타입인 `A` 로 성공할 수도 있다.</br>

`ZIO[R, E, A]` 타입의 값은 다음 함수 타입의 효과적인 버전과 같다:

`R => Either[E, A]`

위 함수는 성공 시 `A`, 실패 시 `E` 인 Either 를 만드는 `R` 을 필요로 한다.</br>
물론 ZIO 이펙트는 실제로 함수는 아니다.</br>
ZIO 이펙트는 동기, 비동기, 동시, 병렬, 풍부한 계산을 할 수 있다.</br>

ZIO 이펙트는 fiber 기반의 동시성 모델을 이용한다. 이는 미리 구축된 스케쥴링, 개별 중단, 구조화된 동시성, 높은 확장성을 제공한다.

`ZIO[R, E, A]` 데이터 타입은 3가지 파라미터를 가진다:

- **`R` - 환경 타입** : 이펙트는 환경 타입인 `R`을 필요로 한다. 만약 환경 타입에 `Any` 타입을 주면, 환경 타입이 이펙트에서 필수가 아니라는 뜻이다. 이펙트를 어떤 값에 대해서도 실행할 수 있기 떄문에(예를 들어, unit 값인 ())
- **`E` - 실패 타입** : 이펙트는 `E` 타입의 값으로 실패할 수 있다. 보통의 어플리케이션에선 `Throwable` 을 사용한다. 만약 실패 타입에 `Nothing` 을 주면, 이 이펙트는 실패할 수 없는 이펙트라는 뜻이다. `Nothing` 타입은 값이 없기 때문에
- **`A` - 성공 타입** : 이펙트는 `A` 타입의 값으로 성공할 수 있다. 만약 성공 타입에 `Unit` 을 주면, 이 이펙트는 유의미한 정보를 만들지 않는다는 뜻이다. 만약 성공 타입에 `Nothing` 을 주면, 이 이펙트는 무한히 실행된다는 뜻이다. (혹은 실패 시까지)

다음 예제에서, `readLine` 함수는 어떤 서비스(환경)도 필요하지 않고, `IOException` 타입으로 실패하거나 `String` 타입으로 성공할 수 있다:
```
import zio._
import java.io.IOException

val readLine: ZIO[Any, IOException, String] =
  Console.readLine
```

`ZIO` 값은 불변이다. 그리고 모든 `ZIO` 함수는 새로운 `ZIO` 값을 만든다. 이는 `ZIO` 를 기존 Scala 의 불변 데이터 구조처럼 추론되고 사용될 수 있게 한다.

`ZIO` 값은 사실상 아무 동작도 하지 않는다. 그저 effectful 한 상호작용을 설명하거나 모델링할 뿐이다.

`ZIO` 는 `ZIO` 실행 시스템에 의해 외부 세계와의 효과적인 상호작용으로 번역된다. </br>
이상적으로, `ZIO` 는 우리의 어플리케이션 `main` 함수에서 단 한번만 수행된다.</br>
`App` 클래스는 이러한 기능을 자동으로 제공한다. 

# Creation
이 장에선 `ZIO` 이펙트를 생성하는 일반적인 방법을 배운다.</br>
`ZIO` 이펙트는 값이나, Scala 타입, 그리고 동기/비동기 부수효과로부터 만들어질 수 있다.

## Success Values
`ZIO.succeed` 함수를 사용하면 특정 값으로부터 `ZIO` 이펙트를 만들어낼 수 있다.
```
import zio._

val s1 = ZIO.succeed(42)
```
`ZIO` 타입 별칭으로 가져올 수도 있다. (`ZIO` 의 `Task` 타입)
```
import zio._

val s2: Task[Int] = ZIO.succeed(42)
```

## Failure Values
`ZIO.fail` 함수를 사용하면 모델의 실패 이펙트를 만들 수 있다:
```
import zio._

val f1 = ZIO.fail("Uh oh!")
```
`ZIO` 데이터 타입은 에러 타입에 대한 제한이 없다. 에러 타입으로 `String`, `Exception`, 혹은 어플리케이션에 적절한 커스텀 타입을 쓸 수도 있다.

많은 어플리케이션은 `Throwable` 이나 `Exception` 으로 모델의 실패를 나타내는데, `ZIO` 로도 가능하다.
```
import zio._

val f2 = ZIO.fail(new Exception("Uh oh!"))
```

## From Values
`ZIO` 는 다양한 데이터 타입을 `ZIO` 이펙트로 바꿀 수 있도록 몇 개의 생성자를 지원한다.

### Option
1. `ZIO.formOption` - `Option` 을 `ZIO` 이펙트로 바꿀 수 있다:
```
import zio._

val zoption: IO[Option[Nothing], Int] = ZIO.fromOption(Some(2))
```
에러 타입인 경우 `Option[Nothing]` 으로 표현된다. 하지만 이는 값이 왜 존재하지 않는지에 대한 정보를 제공하진 않는다.</br>
`mapError` 함수를 사용하면 `Option[Nothing]` 대신 더 정확한 에러 타입으로 나타낼 수도 있다:
```
import zio._

val zoption2: IO[String, Int] = zoption.mapError(_ => "It wasn't there!")
```
결과의 Optional 특성을 유지하면서 다른 연산들과 손쉽게 합성할 수 있다. (이는 `OptionT` 와 유사하다.)
```
import zio._

val maybeId: IO[Option[Nothing], String] = ZIO.fromOption(Some("abc123"))
def getUser(userId: String): IO[Throwable, Option[User]] = ???
def getTeam(teamId: String): IO[Throwable, Team] = ???


val result: IO[Throwable, Option[(User, Team)]] = (for {
  id   <- maybeId
  user <- getUser(id).some
  team <- getTeam(user.teamId).asSomeError
} yield (user, team)).unsome
```
2. `ZIO.some`/`ZIO.none` - Optional 값에 대해 바로 `ZIO` 를 만들 수 있다:
```
import zio._

val someInt: ZIO[Any, Nothing, Option[Int]]     = ZIO.some(3)
val noneInt: ZIO[Any, Nothing, Option[Nothing]] = ZIO.none
```
3. `ZIO.getOrFail` - `Option` 타입을 `ZIO` 환경으로 끌어올릴 수 있다. 만약 option 이 `None` 이라면 적절한 에러 타입으로  `ZIO` 를 실패시킬 수 있다:
- `ZIO.getOrFail` 은 `Throwable` 타입으로 실패시킨다.
- `ZIO.getOrFailUnit` 은 `Unit` 타입으로 실패시킨다.
- `ZIO.getOrFailWith` 는 커스텀 에러 타입으로 실패시킨다.
```
import zio._

def parseInt(input: String): Option[Int] = input.toIntOption

// If the optional value is not defined it fails with Throwable error type:
val r1: ZIO[Any, Throwable, Int] =
  ZIO.getOrFail(parseInt("1.2"))

// If the optional value is not defined it fails with Unit error type:
val r2: ZIO[Any, Unit, Int] =
  ZIO.getOrFailUnit(parseInt("1.2"))

// If the optional value is not defined it fail with given error type:
val r3: ZIO[Any, NumberFormatException, Int] =
  ZIO.getOrFailWith(new NumberFormatException("invalid input"))(parseInt("1.2"))
```
4. `ZIO.noneOrFail` - `Option` 타입을 `ZIO` 값으로 끌어올릴 수 있다. 만약 Option 이 `None` 이라면 `Unit` 타입으로 성공하고, `Some` 이라면 적절한 에러 타입으로 `ZIO` 를 실패시킬 수 있다:
- `ZIO.noneOrFail` 은 `Option` 의 값 타입을 실패시킨다.
- `ZIO.noneOrFailUnit` 은 `Unit` 타입으로 실패시킨다.
- `ZIO.noneOrFailWith` 는 커스텀 에러 타입으로 실패시킨다.
```
import zio._

val optionalValue: Option[String] = ???

// If the optional value is empty it succeeds with Unit
// If the optional value is defined it will fail with the content of the optional value
val r1: ZIO[Any, String, Unit] =
  ZIO.noneOrFail(optionalValue)

// If the optional value is empty it succeeds with Unit
// If the optional value is defined, it will fail by applying the error function to it:
val r2: ZIO[Any, NumberFormatException, Unit] =
  ZIO.noneOrFailWith(optionalValue)(e => new NumberFormatException(e))
```

### Either

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `fromEither` | `Either[E, A]` | `IO[E, A]` |
| `left` | `A` | `UIO[Either[A, Nothing]]` |
| `right` | `A` | `UIO[Either[Nothing, A]]` |

`Either` 는 `ZIO.fromEither` 함수를 사용해 `ZIO` 이펙트로 변환할 수 있다:
```
import zio._

val zeither = ZIO.fromEither(Right("Success!"))
```
에러 타입은 `Either` 의 Left 가 되고, 성공 타입은 Right 가 된다.

### Try

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `fromTry` | `scala.util.Try[A]` | `Task[A]` |

`Try` 값은 `ZIO.fromTry` 함수를 사용해 `ZIO` 이펙트로 변환할 수 있다:
```
import zio._
import scala.util.Try

val ztry = ZIO.fromTry(Try(42 / 0))
```

에러 타입은 항상 `Throwable` 이다. (Try 는 항상 `Throwable` 로 실패하기 때문)

### Future

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `fromFuture` | `ExecutionContext => scala.concurrent.Future[A]` | `Task[A]` |
| `fromFutureJava` | `java.util.concurrent.Future[A]` | `RIO[Blocking, A]` |
| `fromFunctionFuture` | `R => scala.concurrent.Future[A]` | `RIO[R, A]` |
| `fromFutureInterrupt` | `ExecutionContext => scala.concurrent.Future[A]` | `Task[A]` |

`Future` 값은 `ZIO.fromFuture` 함수를 사용해 `ZIO` 이펙트로 변환할 수 있다:
```
import zio._
import scala.concurrent.Future

lazy val future = Future.successful("Hello!")

val zfuture: Task[String] =
  ZIO.fromFuture { implicit ec =>
    future.map(_ => "Goodbye!")
  }
```

`fromFuture` 에 전달된 함수는 `ExecutionContext` 에 전달된다. 이를 통해 `Future` 가 실행되는 Context 를 `ZIO` 가 관리할 수 있다. (물론, `ExecutionContext` 를 무시하는 것도 가능하다).

에러 타입은 항상 `Throwable` 이다. (`Future` 는 항상 `Throwable` 로 실패하기 때문)

### Promise

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `fromPromiseScala` | `scala.concurrent.Promise[A]` | `Task[A]` |

`Promise` 값은 `ZIO.fromPromiseScala` 함수를 사용해 `ZIO` 이펙트로 변환할 수 있다:
```
import zio._
import scala.util._

val func: String => String = s => s.toUpperCase
for {
  promise <- ZIO.succeed(scala.concurrent.Promise[String]())
  _ <- ZIO.attempt {
    Try(func("hello world from future")) match {
      case Success(value) => promise.success(value)
      case Failure(exception) => promise.failure(exception)
    }
  }.fork
  value <- ZIO.fromPromiseScala(promise)
  _ <- Console.printLine(s"Hello World in UpperCase: $value")
} yield ()
```

### Fiber

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `fromFiber` | `Fiber[E, A]` | `IO[E, A]` |
| `fromFiberZIO` | `IO[E, Fiber[E, A]]` | `IO[E, A]` |

`Fiber` 값은 `ZIO.fromFiber` 함수를 사용해 `ZIO` 이펙트로 변환할 수 있다:
```
import zio._

val io: IO[Nothing, String] = ZIO.fromFiber(Fiber.succeed("Hello from Fiber!"))
```

## From Side-Effects

`ZIO` 는 동기/비동기의 부수 효과를 `ZIO` 이펙트로 모두 변환할 수 있다. (순수한 값인)

이 함수들은 절차지향 코드를 감싸는데 이용된다. 이를 통해 `ZIO` 의 모든 기능들을 레거시 Scala, Java 혹은 서드파티 라이브러리 코드에서 사용할 수 있게 된다.

### Synchronous

| **Function** | **Input Type** | **Output Type** | **Note** |
| --- | --- | --- | --- |
| `succeed` | `A` | `UIO[A]` | Imports a total synchronous effect |
| `attempt` | `A` | `Task[A]` | Imports a (partial) synchronous side-effect |

`ZIO.attempt` 를 사용해 동기적인 부수효과를 `ZIO` 이펙트로 변환할 수 있다:
```
import zio._
import scala.io.StdIn

val getLine: Task[String] =
  ZIO.attempt(StdIn.readLine())
```

반환되는 에러 타입은 항상 `Throwable` 이다. 부수효과는 모든 유형의 `Throwable` 로 예외를 발생시킬 수 있기 때문.

부수효과가 어떤 예외도 던지지 않는다면, `ZIO.succeed` 를 사용해 부수효과를 `ZIO` 이펙트로 변환할 수 있다:
```
import zio._

def printLine(line: String): UIO[Unit] =
  ZIO.succeed(println(line))

val succeedTask: UIO[Long] =
  ZIO.succeed(java.lang.System.nanoTime())
```
모든 부수효과에 대해 확실하게 알고 있지 않다면 `ZIO.succeed` 대신 `ZIO.attempt` 를 사용해 이펙트로 변환하는게 좋다.

너무 광범위하다면, `ZIO.retainToOrDie` 를 사용해 특정 예외에 대해서는 유지시키고 다른 예외에 대해선 `ZIO` 를 끝낼 수 있다.
```
import zio._
import java.io.IOException

val printLine2: IO[IOException, String] =
  ZIO.attempt(scala.io.StdIn.readLine()).refineToOrDie[IOException]
```

### Blocking Synchronous Side-Effects

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `blocking` | `ZIO[R, E, A]` | `ZIO[R, E, A]` |
| `attemptBlocking` | `A` | `RIO[Blocking, A]` |
| `attemptBlockingCancelable` | `effect: => A`,`cancel: UIO[Unit]` | `RIO[Blocking, A]` |
| `attemptBlockingInterrupt` | `A` | `RIO[Blocking, A]` |
| `attemptBlockingIO` | `A` | `ZIO[Blocking, IOException, A]` |

어떤 부수효과들은 IO 를 blocking 하거나 쓰레드를 대기 상태로 만든다. 이러한 부수효과들을 잘 관리하지 않으면 메인 스레드 풀이 고갈되어 작업이 지연될 수 있다.

`ZIO` 는 이러한 blocking 부수효과들을 안전하게 `ZIO` 이펙트로 변환할 수 있는 `zio.blocking` 패키지를 제공한다.

`attemptBlocking` 을 사용해 blocking 부수효과를 `ZIO` 이펙트로 변환할 수 있다.
```
import zio._

val sleeping =
  ZIO.attemptBlocking(Thread.sleep(Long.MaxValue))
```

결과 이펙트는 blocking 이펙트를 위해 특별히 설계된 별도의 스레드 풀에서 수행된다.

`attemptBlockingInterrupt` 를 사용하면 `Thread.interrupt` 를 호출해 blocking 부수효과를 중단시킬 수 있다.

어떤 blocking 부수효과들은 취소 이펙트 호출로만 중단될 수 있다.</br>
`attemptBlockingCancelable` 를 사용하면 이러한 부수효과를 변환할 수 있다:
```
import zio._
import java.net.ServerSocket

def accept(l: ServerSocket) =
  ZIO.attemptBlockingCancelable(l.accept())(ZIO.succeed(l.close()))
```

부수 효과가 이미 `ZIO` 이펙트로 변환되었다면, `attemptBlocking` 대신 `blocking` 메소드를 사용해 blocking 스레드 풀에서 이펙트가 수행됨을 보장할 수 있다:</br>
```
import zio._
import scala.io.{ Codec, Source }

def download(url: String) =
  ZIO.attempt {
    Source.fromURL(url)(Codec.UTF8).mkString
  }

def safeDownload(url: String) =
  ZIO.blocking(download(url))
```

### Asynchronous

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `async` | `(ZIO[R, E, A] => Unit) => Any` | `ZIO[R, E, A]` |
| `asyncZIO` | `(ZIO[R, E, A] => Unit) => ZIO[R, E, Any]` | `ZIO[R, E, A]` |
| `asyncMaybe` | `(ZIO[R, E, A] => Unit) => Option[ZIO[R, E, A]]` | `ZIO[R, E, A]` |
| `asyncInterrupt` | `(ZIO[R, E, A] => Unit) => Either[URIO[R, Any], ZIO[R, E, A]]` | `ZIO[R, E, A]` |

`ZIO.async` 를 사용해 콜백기반 API 의 비동기 부수효과를 `ZIO` 이펙트로 변환할 수 있다:
```
import zio._

object legacy {
  def login(
    onSuccess: User => Unit,
    onFailure: AuthError => Unit): Unit = ???
}

val login: IO[AuthError, User] =
  ZIO.async[Any, AuthError, User] { callback =>
    legacy.login(
      user => callback(ZIO.succeed(user)),
      err  => callback(ZIO.fail(err))
    )
  }
```

비동기 `ZIO` 이펙트는 콜백기반 API 들보다 훨씬 쉽고, 인터럽션/자원 안전성/우수한 에러 처리 같은 `ZIO` 의 장점을 가질 수 있다.

## Creating Suspended Effects

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `suspend` | `RIO[R, A]` | `RIO[R, A]` |
| `suspendSucceed` | `ZIO[R, E, A]` | `ZIO[R, E, A]` |

`suspend` 메소드를 사용해 `RIO[R, A]` 이펙트를 정지시킬 수 있다:
```
import zio._
import java.io.IOException

val suspendedEffect: RIO[Any, ZIO[Any, IOException, Unit]] =
  ZIO.suspend(ZIO.attempt(Console.printLine("Suspended Hello World!")))
```

# Blocking Operations

`ZIO` 는 blocking 명령(스레드 sleep, 동기 소켓/파일 읽기 등등)을 수행할 때 사용할 수 있는 스레드 풀에 대한 접근을 제공한다.

기본적으로 `ZIO ` 는 비동기이며 모든 이펙트는 비동기 명령 최적화가 된 기본 primary 스레드에서 수행된다.</br>
`ZIO` 는 fiber 기반의 동시성 모델을 사용하므로, primary 스레드 풀에서 I/O 를 blocking 하거나 CPU 작업을 수행하는 경우 모든 primary 스레드 풀을 독점하게 된다.

다음 예제에서 비동기 primary 스레드 풀에서 병렬로 수행할 20 개의 blocking 작업을 만들었다.</br>
우리의 머신이 8 CPU 코어라면 `ZIO` 는 16 (2 * 8) 사이즈의 스레드 풀을 만들 것이다. 이 프로그램을 돌리면 모든 스레드가 막히고 4 개(20 - 16)의 blocking 작업이 남을 것이다.
```
import zio._

def blockingTask(n: Int): UIO[Unit] =
  Console.printLine(s"running blocking task number $n").orDie *>
    ZIO.succeed(Thread.sleep(3000)) *>
    blockingTask(n)

val program = ZIO.foreachPar((1 to 100).toArray)(blockingTask)
```

## Creating Blocking Effects

`ZIO` 는 blocking I/O, CPU 작업을 위해 특별히 설게된 별도의 스레드 풀을 가진다. </br>
우리는 primary 스레드 풀에 대한 간섭을 방지하기 위해 blocking 작업을 이 스레드 풀에서 수행해야 한다.

스레드 풀은 무제한의 작업을 허용하고(메모리가 가능할 때까지), 필요에 따라 새로운 스레드를 계속 만든다.

`blocking` 명령은 `ZIO` 이펙트를 받아 blocking 스레드 풀에서 실행할 다른 이펙트를 반환한다.

또한, `attemptBlocking` 을 사용해 I/O 를 blocking 하는 동기적인 이펙트를 `ZIO` 로 변환할 수 있다:
```
import zio._

def blockingTask(n: Int) = ZIO.attemptBlocking {
  do {
    println(s"Running blocking task number $n on dedicated blocking thread pool")
    Thread.sleep(3000)
  } while (true)
}
```

# Mapping

## map

`A => B` 형태의 함수를 사용하면 `IO[E, A]` 타입을 `IO[E, B]` 로 바꿀 수 있다. 이를 통해 작업에 의해 생성된 값을 다른 값으로 변환할 수 있다.
```
import zio._

val mappedValue: UIO[Int] = ZIO.succeed(21).map(_ * 2)
```
## Tapping

`ZIO.tap` 을 사용하면 기존 이펙트의 반환값을 바꾸지 않고 성공 값으로 다른 effectful 한 작업을 할 수 있다. (기존 ZIO 의 성공 값을 중간에 볼 수 있다.)
```
trait ZIO[-R, +E, +A] {
  def tap[R1 <: R, E1 >: E](f: A => ZIO[R1, E1, Any]): ZIO[R1, E1, A]
  def tapSome[R1 <: R, E1 >: E](f: PartialFunction[A, ZIO[R1, E1, Any]]): ZIO[R1, E1, A]
}

import zio._

import java.io.IOException

object MainApp extends ZIOAppDefault {
  def isPrime(n: Int): Boolean =
    if (n <= 1) false else (2 until n).forall(i => n % i != 0)

  val myApp: ZIO[Any, IOException, Unit] =
    for {
      ref <- Ref.make(List.empty[Int])
      prime <-
        Random
          .nextIntBetween(0, Int.MaxValue)
          .tap(random => ref.update(_ :+ random))
          .repeatUntil(isPrime)
      _ <- Console.printLine(s"found a prime number: $prime")
      tested <- ref.get
      _ <- Console.printLine(
        s"list of tested numbers: ${tested.mkString(", ")}"
      )
    } yield ()

  def run = myApp
}
```

## Chaining

`flatMap` 을 사용하면 두 가지 작업을 차례대로 수행할 수 있다. 첫 번째 작업은 두 번째 작업에 영향을 줄 수 있다.
```
import zio._

val chainedActionsValue: UIO[List[Int]] = ZIO.succeed(List(1, 2, 3)).flatMap { list =>
  ZIO.succeed(list.map(_ + 1))
}

```

첫 번째 작업이 실패하면 `flatMap` 을 통과한 callback 은 수행되지 않는다. 따라서 `flatMap` 에 의해 반환된 합성 이펙트도 실패한다.

이펙트의 모든 체이닝에서, 첫 실패는 모든 체이닝을 실패시킨다.(short-circuit)</br>
(예외를 던지면 나머지 명령문이 조기 종료되는 것과 같이)

`ZIO` 데이터 타입이 `flatMap` 과 `map` 을 지원하기 때문에 scala 의 for comprehensions 를 이용해 순차 이펙트를 만들 수 있다:
```
import zio._

val program =
  for {
    _    <- Console.printLine("Hello! What is your name?")
    name <- Console.readLine
    _    <- Console.printLine(s"Hello, ${name}, welcome to ZIO!")
  } yield ()
```
for comprehensions 은 일련의 이펙트를 구성하기 위한 보다 절차적인 문법을 제공한다.

## Zipping

`zip` 을 사용하면 두 개의 이펙트를 하나로 합칠 수 있다. zip 된 이펙트는 두 이펙트가 성공한 값의 튜플로 성공한다.
```
import zio._

val zipped: UIO[(String, Int)] =
  ZIO.succeed("4").zip(ZIO.succeed(2))
```

`zip` 은 항상 순차적으로 수행되기 때문에, 왼쪽 이펙트가 오른쪽보다 **먼저** 수행된다.

모든 `zip` 명령은 왼쪽이나 오른쪽 이펙트가 하나라도 실패하면 실패한다. 양쪽의 성공 값이 필요하기 때문.

### zipLeft and zipRight

때때로 이펙트의 성공 값이 유용하지 않을 때가 있다.(에를 들어 `Unit`)</br>
이런 경우, `zipLeft` 나 `zipRight` 를 사용하면 편리하다.</br>
이 함수들은 처음 `zip` 을 수행하고 이후엔 튜플에 `map` 을 수행하며 나머지 한 쪽을 무시한다:
```
import zio._

val zipRight1 =
  Console.printLine("What is your name?").zipRight(Console.readLine)
```

`zipLeft` 와 `zipRight` 는 각각 `*>` , `<*` 연산자로 사용할 수도 있다.
```
import zio._

val zipRight2 =
  Console.printLine("What is your name?") *>
  Console.readLine
```

# Parallelism

`ZIO` 는 이펙트를 병렬적으로 수행하기 위한 많은 명령을 제공한다.</br>
이 함수들은 각 함수 이름에 `Par` 접미사가 뒤에 붙는다.(Parallelism 의 Par)

예를 들어, `ZIO#zip` 함수는 두 이펙트를 순차적으로 묶는다. 하지만 `ZIO#zipPar` 함수는 병렬적으로 두 이펙트를 묶는다.

다음 표는 각각의 함수를 parallel 버전과 매핑했다:

| **Description** | **Sequential** | **Parallel** |
| --- | --- | --- |
| Zip two effects into one | `ZIO#zip` | `ZIO#zipPar` |
| Zip two effects into one | `ZIO#zipWith` | `ZIO#zipWithPar` |
| Collect from many effects | `ZIO.collectAll` | `ZIO.collectAllPar` |
| Effectfully loop over values | `ZIO.foreach` | `ZIO.foreachPar` |
| Reduce many values | `ZIO.reduceAll` | `ZIO.reduceAllPar` |
| Merge many values | `ZIO.mergeAll` | `ZIO.mergeAllPar` |

불필요한 연산을 줄이기 위해, 모든 병렬 명령에선 한 이펙트가 실패하면 다른 이펙트들은 중단된다.

fail-fast 방식이 싫다면, 잠재적으로 실패한 이펙트는 `ZIO#either` 나 `ZIO#option` 을 사용하여 오류없는 이펙트로 변환할 수 있다.

## Racing

`ZIO` 의 `race` 함수는 여러 개의 병렬 이펙트 중 가장 먼저 성공한 결과 값을 반환한다:
```
import zio._

for {
  winner <- ZIO.succeed("Hello").race(ZIO.succeed("Goodbye"))
} yield winner
```

첫 번째 성공보다 첫 번째 (성공 or 실패) 의 조합을 얻고 싶다면, `left.either race right.either` 와 같이 하면 된다.

## Timeout

`ZIO#timeout` 을 사용해 모든 이펙트에 대해 timeout 을 걸 수 있다. 이는 `Option` 값으로 성공하는데, `None` 이라면 이펙트가 완료되기 전에 timeout 이 난 것이다.
```
import zio._

ZIO.succeed("Hello").timeout(10.seconds)
```

이펙트가 timeout 되면 백그라운드에서 작업을 계속하는 대신 중단시켜 자원이 낭비되지 않는다.

# Error Management
## Either

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `ZIO#either` | `ZIO[R, E, A]` | `ZIO#zipPar` |
| `ZIO.absolve` | `ZIO[R, E, Either[E, A]]` | `ZIO[R, E, A]` |

`ZIO#either` 는 `ZIO[R, E, A]` 를 받고 `ZIO[R, Nothing, Either[E, A]]` 를 생성함으로써 에러를 뽑아낼 수 있다.
```
val zeither: UIO[Either[String, Int]] =
  ZIO.fail("Uh oh!").either
```

`either` 와 반대되는 `ZIO.absolve` 는 `ZIO[R, Nothing, Either[E, A]]` 를 받고 `ZIO[R, E, A]` 를 생성함으로써 에러를 뽑아낼 수 있다.
```
def sqrt(io: UIO[Double]): IO[String, Double] =
  ZIO.absolve(
    io.map(value =>
      if (value < 0.0) Left("Value must be >= 0.0")
      else Right(Math.sqrt(value))
    )
  )
```

## Catching

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `ZIO#catchAll` | `E => ZIO[R1, E2, A1]` | `ZIO[R1, E2, A1]` |
| `ZIO#catchAllCause` | `Cause[E] => ZIO[R1, E2, A1]` | `ZIO[R1, E2, A1]` |
| `ZIO#catchAllDefect` | `Throwable => ZIO[R1, E1, A1]` | `ZIO[R1, E1, A1]` |
| `ZIO#catchAllTrace` | `((E, Option[StackTrace])) => ZIO[R1, E2, A1]` | `ZIO[R1, E2, A1]` |
| `ZIO#catchSome` | `PartialFunction[E, ZIO[R1, E1, A1]]` | `ZIO[R1, E1, A1]` |
| `ZIO#catchSomeCause` | `PartialFunction[Cause[E], ZIO[R1, E1, A1]]` | `ZIO[R1, E1, A1]` |
| `ZIO#catchSomeDefect` | `PartialFunction[Throwable, ZIO[R1, E1, A1]]` | `ZIO[R1, E1, A1]` |
| `ZIO#catchSomeTrace` | `PartialFunction[(E, Option[StackTrace]), ZIO[R1, E1, A1]]` | `ZIO[R1, E1, A1]` |

### Catching All Errors

`catchAll` 을 사용하면 **모든** 타입의 에러를 recover 및 catch 를 할 수 있다:
```
val z: IO[IOException, Array[Byte]] =
  readFile("primary.json").catchAll(_ =>
    readFile("backup.json"))
```

`catchAll` 을 통과하는 콜백에서 다른 에러 타입을 가지는 이펙트를 반환할 수 있다.(혹은 `Nothing`)</br>
이는 `catchAll` 의 반환 이펙트에 반영된다.

### Catching Some Errors

`catchSome` 을 사용하면 **특정** 타입의 에러를 recover 및 catch 를 할 수 있다:
```
val data: IO[IOException, Array[Byte]] =
  readFile("primary.data").catchSome {
    case _ : FileNotFoundException =>
      readFile("backup.data")
  }
```

`catchAll` 과는 다르게 `catchSome` 은 오류 유형을 더 넓은 범위로 확장하는 것만 가능하다.

## Fallback

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `orElse` | `ZIO[R1, E2, A1]` | `ZIO[R1, E2, A1]` |
| `orElseEither` | `ZIO[R1, E2, B]` | `ZIO[R1, E2, Either[A, B]]` |
| `orElseFail` | `E1` | `ZIO[R, E1, A]` |
| `orElseOptional` | `ZIO[R1, Option[E1], A1]` | `ZIO[R1, Option[E1], A1]` |
| `orElseSucceed` | `A1` | `URIO[R, A1]` |

`orElse` 의 조합을 통해 이펙트가 실패했을 때 다른 이펙트를 시도해 나갈 수 있다:
```
val primaryOrBackupData: IO[IOException, Array[Byte]] =
  readFile("primary.data").orElse(readFile("backup.data"))
```

## Folding

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `fold` | `failure: E => B, success: A => B` | `URIO[R, B]` |
| `foldCause` | `failure: Cause[E] => B, success: A => B` | `URIO[R, B]` |
| `orElseFail` | `failure: E => ZIO[R1, E2, B], success: A => ZIO[R1, E2, B]` | `ZIO[R1, E2, B]` |
| `orElseOptional` | `failure: Cause[E] => ZIO[R1, E2, B], success: A => ZIO[R1, E2, B]` | `ZIO[R1, E2, B]` |
| `orElseSucceed` | `failure: ((E, Option[StackTrace])) => ZIO[R1, E2, B], success: A => ZIO[R1, E2, B]` | `ZIO[R1, E2, B]` |

Scala 의 `Option` 과 `Either` 는 동시에 실패, 성공을 처리할 수 있는 `fold` 를 제공한다.</br>
이와 비슷하게, `ZIO` 도 실패, 성공을 동시에 처리하는 몇몇의 함수를 제공한다.

첫째로 `fold` 함수는 실패, 성공에 대한 처리를 non-effectful 하게 할 수 있도록 제공한다:(non-effectful 한 핸들러를 제공한다.)
```
lazy val DefaultData: Array[Byte] = Array(0, 0)

val primaryOrDefaultData: UIO[Array[Byte]] =
  readFile("primary.data").fold(
    _    => DefaultData,
    data => data)
```

두 번째로 `foldZio` 함수는 실패, 성공에 대한 처리를 effectful 하게 할 수 있도록 제공한다:(effectful 한 핸들러를 제공한다.)
```
val primaryOrSecondaryData: IO[IOException, Array[Byte]] =
  readFile("primary.data").foldZIO(
    _    => readFile("secondary.data"),
    data => ZIO.succeed(data))
```

`foldZio` 는 강력하고 빠르기 때문에 거의 모든 에러 처리 함수에서 정의해 사용한다.

아래 예제는 `readUrls` 함수의 실패, 성공을 `foldZIO` 로 처리하는 예제이다:
```
val urls: UIO[Content] =
  readUrls("urls.json").foldZIO(
    error   => ZIO.succeed(NoContent(error)),
    success => fetchContent(success)
  )
```

## Retrying

| **Function** | **Input Type** | **Output Type** |
| --- | --- | --- |
| `retry` | `Schedule[R1, E, S]` | `ZIO[R1, E, A]` |
| `retryN` | `n: Int` | `ZIO[R, E, A]` |
| `retryOrElse` | `policy: Schedule[R1, E, S], orElse: (E, S) => ZIO[R1, E1, A1]` | `ZIO[R1, E1, A1]` |
| `retryOrElseEither` | `schedule: Schedule[R1, E, Out], orElse: (E, Out) => ZIO[R1, E1, B]` | `ZIO[R1, E1, Either[B, A]]` |
| `retryUntil` | `E => Boolean` | `ZIO[R, E, A]` |
| `retryUntilEquals` | `E1` | `ZIO[R, E1, A]` |
| `retryUntilZIO` | `E => URIO[R1, Boolean]` | `ZIO[R1, E, A]` |
| `retryWhile` | `E => Boolean` | `ZIO[R, E, A]` |
| `retryWhileEquals` | `E1` | `ZIO[R, E1, A]` |
| `retryWhileZIO` | `E => URIO[R1, Boolean]` | `ZIO[R1, E, A]` |

어플리케이션을 구축할 땐 과도한 에러에 대한 복원력이 필요한데, 이를 위해 에러에 대한 재시도가 필요하다.

실패 이펙트를 재시도하기 위한 유용한 `ZIO` 함수들이 있다.

가장 기본인 `ZIO#retry` 함수는, 특정 정책에 의해 이펙트가 실패하면 `Schedule` 을 받아 첫 이펙트를 재시도하는 새로운 이펙트를 반환한다:
```
val retriedOpenFile: ZIO[Any, IOException, Array[Byte]] =
  readFile("primary.data").retry(Schedule.recurs(5))
```

다음으로 가장 강력한 함수인 `ZIO#retryOrElse` 는, 특정 정책에 의해 이펙트가 실패했을 때의 대체 fallback 을 정의할 수 있다:
```
readFile("primary.data").retryOrElse(
  Schedule.recurs(5),
  (_, _:Long) => ZIO.succeed(DefaultData)
)
```

마지막으로 `ZIO#retryOrElseEither` 함수는, 다른 타입의 fallback 을 반환할 수 있다.(Either 로 감싼)

# Resource Management

ZIO's resource management features work across synchronous, asynchronous, concurrent, and other effect types, and provide strong guarantees even in the presence of failure, interruption, or defects in the application.

Finalizing
Scala has a try / finally construct which helps us to make sure we don't leak resources because no matter what happens in the try, the finally block will be executed. So we can open files in the try block, and then we can close them in the finally block, and that gives us the guarantee that we will not leak resources.

Asynchronous Try / Finally
The problem with the try / finally construct is that it only applies to synchronous code, meaning it doesn't work for asynchronous code. ZIO gives us a method called ensuring that works with either synchronous or asynchronous actions. So we have a functional try / finally even for asynchronous regions of our code.

Like try / finally, the ensuring operation guarantees that if an effect begins executing and then terminates (for whatever reason), then the finalizer will begin executing:

import zio._

val finalizer =
  ZIO.succeed(println("Finalizing!"))

val finalized: IO[String, Unit] =
  ZIO.fail("Failed!").ensuring(finalizer)

The finalizer is not allowed to fail, which means that it must handle any errors internally.

Like try / finally, finalizers can be nested, and the failure of any inner finalizer will not affect outer finalizers. Nested finalizers will be executed in reverse order, and linearly (not in parallel).

Unlike try / finally, ensuring works across all types of effects, including asynchronous and concurrent effects.

Here is another example of ensuring that our clean-up action is called before our effect is done:

import zio._

import zio.Task
var i: Int = 0
val action: Task[String] =
  ZIO.succeed(i += 1) *>
    ZIO.fail(new Throwable("Boom!"))
val cleanupAction: UIO[Unit] = ZIO.succeed(i -= 1)
val composite = action.ensuring(cleanupAction)

CAUTION
Finalizers offer very powerful guarantees, but they are low-level, and should generally not be used for releasing resources. For higher-level logic built on ensuring, see ZIO#acquireReleaseWith in the acquire release section.

Unstoppable Finalizers
In Scala when we nest try / finally finalizers, they cannot be stopped. If we have nested finalizers and one of them fails for some sort of catastrophic reason the ones on the outside will still be run and in the correct order.

try {
  try {
    try {
      ...
    } finally f1
  } finally f2
} finally f3

Also in ZIO like try / finally, the finalizers are unstoppable. This means if we have a buggy finalizer that is going to leak some resources, we will leak the minimum amount of resources because all other finalizers will still be run in the correct order.

val io = ???
io.ensuring(f1)
 .ensuring(f2)
 .ensuring(f3)

AcquireRelease
In Scala the try / finally is often used to manage resources. A common use for try / finally is safely acquiring and releasing resources, such as new socket connections or opened files:

val handle = openFile(name)

try {
  processFile(handle)
} finally closeFile(handle)

ZIO encapsulates this common pattern with ZIO#acquireRelease, which allows us to specify an acquire effect, which acquires a resource; a release effect, which releases it; and a use effect, which uses the resource. Acquire release lets us open a file and close the file and no matter what happens when we are using that resource.

The release action is guaranteed to be executed by the runtime system, even if the utilize action throws an exception or the executing fiber is interrupted.

Acquire release is a built-in primitive that let us safely acquire and release resources. It is used for a similar purpose as try / catch / finally, only acquire release work with synchronous and asynchronous actions, work seamlessly with fiber interruption, and is built on a different error model that ensures no errors are ever swallowed.

Acquire release consist of an acquire action, a utilize action (which uses the acquired resource), and a release action.

import zio._

val groupedFileData: IO[IOException, Unit] = ZIO.acquireReleaseWith(openFile("data.json"))(closeFile(_)) { file =>
  for {
    data    <- decodeData(file)
    grouped <- groupData(data)
  } yield grouped
}


Acquire releases have compositional semantics, so if an acquire release is nested inside another acquire release, and the outer resource is acquired, then the outer release will always be called, even if, for example, the inner release fails.

Let's look at a full working example on using acquire release:

import zio._
import java.io.{ File, FileInputStream }
import java.nio.charset.StandardCharsets

object Main extends ZIOAppDefault {

  // run my acquire release
  def run = myAcquireRelease

  def closeStream(is: FileInputStream) =
    ZIO.succeed(is.close())

  def convertBytes(is: FileInputStream, len: Long) =
    ZIO.attempt {
      val buffer = new Array[Byte](len.toInt)
      is.read(buffer)
      println(new String(buffer, StandardCharsets.UTF_8))
    }

  // myAcquireRelease is just a value. Won't execute anything here until interpreted
  val myAcquireRelease: Task[Unit] = for {
    file   <- ZIO.attempt(new File("/tmp/hello"))
    len    = file.length
    string <- ZIO.acquireReleaseWith(ZIO.attempt(new FileInputStream(file)))(closeStream)(convertBytes(_, len))
  } yield string
}


ZIO Aspect
There are two types of concerns in an application, core concerns, and cross-cutting concerns. Cross-cutting concerns are shared among different parts of our application. We usually find them scattered and duplicated across our application, or they are tangled up with our primary concerns. This reduces the level of modularity of our programs.

A cross-cutting concern is more about how we do something than what we are doing. For example, when we are downloading a bunch of files, creating a socket to download each one is the core concern because it is a question of what rather than the how, but the following concerns are cross-cutting ones:

Downloading files sequentially or in parallel
Retrying and timing out the download process
Logging and monitoring the download process
So they don't affect the return type of our workflows, but they add some new aspects or change their behavior.

To increase the modularity of our applications, we can separate cross-cutting concerns from the main logic of our programs. ZIO supports this programming paradigm, which is called aspect-oriented programming.

The ZIO effect has a data type called ZIOAspect, which allows modifying a ZIO effect and convert it into a specialized ZIO effect. We can add a new aspect to a ZIO effect with @@ syntax like this:

import zio._

val myApp: ZIO[Any, Throwable, String] =
  ZIO.attempt("Hello!") @@ ZIOAspect.debug

As we see, the debug aspect doesn't change the return type of our effect, but it adds a new debugging aspect to our effect.

ZIOAspect is like a transformer of the ZIO effect, which takes a ZIO effect and converts it to another ZIO effect. We can think of a ZIOAspect as a function of type ZIO[R, E, A] => ZIO[R, E, A].

To compose multiple aspects, we can use @@ operator:

import zio._

def download(url: String): ZIO[Any, Throwable, Chunk[Byte]] = ZIO.succeed(???)

ZIO.foreachPar(List("zio.dev", "google.com")) { url =>
  download(url) @@
    ZIOAspect.retry(Schedule.fibonacci(1.seconds)) @@
    ZIOAspect.loggedWith[Chunk[Byte]](file => s"Downloaded $url file with size of ${file.length} bytes")
}


The order of aspect composition matters. Therefore, if we change the order, the behavior may change.