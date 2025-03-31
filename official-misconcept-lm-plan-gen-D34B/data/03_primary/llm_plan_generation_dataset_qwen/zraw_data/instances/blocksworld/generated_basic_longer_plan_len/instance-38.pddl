

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(ontable b)
(ontable c)
(on d a)
(on e d)
(on f e)
(clear c)
(clear f)
)
(:goal
(and
(on b c)
(on c f)
(on d b)
(on e d)
(on f a))
)
)


