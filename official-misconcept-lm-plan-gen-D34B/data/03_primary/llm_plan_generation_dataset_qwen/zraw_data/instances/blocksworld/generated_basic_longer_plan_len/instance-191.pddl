

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(ontable a)
(on b g)
(on c e)
(on d c)
(on e f)
(on f h)
(on g d)
(on h i)
(on i j)
(ontable j)
(clear a)
(clear b)
)
(:goal
(and
(on b c)
(on c f)
(on d h)
(on e i)
(on f j)
(on h b)
(on j a))
)
)


