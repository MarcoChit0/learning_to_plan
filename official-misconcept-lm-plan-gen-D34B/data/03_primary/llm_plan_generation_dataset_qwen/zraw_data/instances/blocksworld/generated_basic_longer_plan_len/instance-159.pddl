

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(on b c)
(ontable c)
(on d b)
(on e g)
(on f d)
(ontable g)
(ontable h)
(on i e)
(clear a)
(clear f)
(clear i)
)
(:goal
(and
(on a h)
(on b a)
(on c f)
(on e c)
(on f g)
(on g d)
(on h i)
(on i e))
)
)


