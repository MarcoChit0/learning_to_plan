

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(on b c)
(on c i)
(on d a)
(ontable e)
(on f d)
(ontable g)
(on h b)
(on i g)
(clear e)
(clear f)
)
(:goal
(and
(on a c)
(on b a)
(on c f)
(on d g)
(on e i)
(on f d)
(on h b)
(on i h))
)
)


