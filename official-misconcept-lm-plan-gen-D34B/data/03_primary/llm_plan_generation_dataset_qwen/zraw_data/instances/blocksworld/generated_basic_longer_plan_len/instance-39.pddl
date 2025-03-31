

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b a)
(on c f)
(on d e)
(on e b)
(on f d)
(clear c)
)
(:goal
(and
(on a c)
(on b d)
(on c e)
(on d a)
(on e f))
)
)


