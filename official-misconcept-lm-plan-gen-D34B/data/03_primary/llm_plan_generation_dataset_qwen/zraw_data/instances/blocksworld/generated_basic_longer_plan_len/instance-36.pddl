

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b d)
(ontable c)
(on d f)
(on e a)
(on f e)
(clear b)
(clear c)
)
(:goal
(and
(on b c)
(on c d)
(on d e)
(on e f)
(on f a))
)
)


