

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(on b d)
(on c a)
(on d e)
(on e f)
(ontable f)
(clear c)
)
(:goal
(and
(on a b)
(on b c)
(on c e)
(on e d)
(on f a))
)
)


