

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a c)
(on b g)
(ontable c)
(on d a)
(on e d)
(on f b)
(ontable g)
(clear e)
(clear f)
)
(:goal
(and
(on a c)
(on b d)
(on c g)
(on d f)
(on e a)
(on f e))
)
)


