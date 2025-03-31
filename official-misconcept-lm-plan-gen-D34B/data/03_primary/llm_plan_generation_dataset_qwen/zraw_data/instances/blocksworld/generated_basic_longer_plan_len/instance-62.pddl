

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a d)
(on b g)
(on c e)
(on d b)
(on e a)
(ontable f)
(on g f)
(clear c)
)
(:goal
(and
(on a f)
(on b a)
(on c g)
(on e b)
(on f d))
)
)


