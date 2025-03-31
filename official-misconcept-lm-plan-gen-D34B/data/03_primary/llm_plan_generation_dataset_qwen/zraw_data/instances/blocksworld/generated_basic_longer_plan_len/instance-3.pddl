

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(on b a)
(on c f)
(ontable d)
(on e d)
(ontable f)
(clear b)
(clear c)
)
(:goal
(and
(on a e)
(on b d)
(on c f)
(on e c)
(on f b))
)
)


