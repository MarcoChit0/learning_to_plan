

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a c)
(on b a)
(ontable c)
(on d e)
(on e f)
(on f b)
(on g d)
(clear g)
)
(:goal
(and
(on a b)
(on b d)
(on d e)
(on e f)
(on f c))
)
)


