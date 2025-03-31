

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b a)
(on c e)
(on d c)
(on e b)
(on f d)
(clear f)
)
(:goal
(and
(on a c)
(on d e)
(on e b)
(on f a))
)
)


