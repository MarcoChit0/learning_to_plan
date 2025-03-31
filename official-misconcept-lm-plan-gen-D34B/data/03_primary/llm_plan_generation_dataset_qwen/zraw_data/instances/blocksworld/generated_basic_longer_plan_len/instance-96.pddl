

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a f)
(on b c)
(on c d)
(on d a)
(on e b)
(ontable f)
(ontable g)
(on h g)
(clear e)
(clear h)
)
(:goal
(and
(on a c)
(on b a)
(on c g)
(on e h)
(on g f)
(on h b))
)
)


