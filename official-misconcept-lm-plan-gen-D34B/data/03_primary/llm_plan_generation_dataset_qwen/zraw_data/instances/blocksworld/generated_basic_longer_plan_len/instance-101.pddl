

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b g)
(on c f)
(on d b)
(on e d)
(on f h)
(ontable g)
(on h e)
(clear a)
(clear c)
)
(:goal
(and
(on a f)
(on b c)
(on c e)
(on e a)
(on f g)
(on h b))
)
)


