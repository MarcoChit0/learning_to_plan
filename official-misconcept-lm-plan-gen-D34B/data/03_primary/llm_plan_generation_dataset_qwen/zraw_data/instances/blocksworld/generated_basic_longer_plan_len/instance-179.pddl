

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a b)
(on b g)
(on c i)
(on d e)
(on e f)
(on f c)
(ontable g)
(on h d)
(on i a)
(clear h)
)
(:goal
(and
(on b g)
(on c h)
(on e i)
(on f e)
(on g d)
(on h b)
(on i a))
)
)


