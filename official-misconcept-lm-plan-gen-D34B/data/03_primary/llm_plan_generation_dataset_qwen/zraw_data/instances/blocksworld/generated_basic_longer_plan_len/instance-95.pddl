

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a e)
(on b h)
(on c d)
(on d f)
(on e c)
(on f g)
(ontable g)
(on h a)
(clear b)
)
(:goal
(and
(on a f)
(on b g)
(on c a)
(on d e)
(on g d)
(on h b))
)
)


