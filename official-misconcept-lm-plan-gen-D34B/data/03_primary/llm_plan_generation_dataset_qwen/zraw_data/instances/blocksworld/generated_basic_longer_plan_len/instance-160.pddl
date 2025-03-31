

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a b)
(ontable b)
(on c g)
(on d h)
(on e d)
(on f i)
(on g f)
(on h c)
(on i a)
(clear e)
)
(:goal
(and
(on a c)
(on c i)
(on d e)
(on e b)
(on f d)
(on h a)
(on i f))
)
)


