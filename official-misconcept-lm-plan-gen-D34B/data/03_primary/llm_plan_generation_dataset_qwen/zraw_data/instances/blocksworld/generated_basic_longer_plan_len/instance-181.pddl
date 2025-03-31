

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b i)
(ontable c)
(on d h)
(on e b)
(on f a)
(on g d)
(on h c)
(on i g)
(clear e)
(clear f)
)
(:goal
(and
(on a h)
(on b c)
(on c g)
(on d f)
(on e i)
(on f a)
(on g e)
(on i d))
)
)


