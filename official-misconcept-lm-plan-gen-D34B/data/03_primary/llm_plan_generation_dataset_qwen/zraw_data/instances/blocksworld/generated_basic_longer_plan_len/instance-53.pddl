

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a e)
(on b f)
(ontable c)
(on d g)
(ontable e)
(on f a)
(on g b)
(clear c)
(clear d)
)
(:goal
(and
(on a e)
(on b d)
(on c f)
(on d g)
(on e b)
(on g c))
)
)


