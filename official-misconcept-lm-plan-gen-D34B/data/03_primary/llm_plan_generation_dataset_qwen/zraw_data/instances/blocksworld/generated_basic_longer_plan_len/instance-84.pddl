

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a d)
(ontable b)
(on c f)
(on d c)
(on e a)
(on f g)
(on g b)
(clear e)
)
(:goal
(and
(on a b)
(on b d)
(on c e)
(on e f)
(on g a))
)
)


