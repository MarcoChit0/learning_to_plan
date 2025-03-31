(define (problem prob)
 (:domain barman)
 (:objects 
      shaker368 - shaker
      left right - hand
      shot280 shot231 - shot
      ingredient57 ingredient38 ingredient59 ingredient328 - ingredient
      cocktail439 - cocktail
      dispenser360 dispenser65 dispenser292 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker368)
  (ontable shot280)
  (ontable shot231)
  (dispenses dispenser360 ingredient57)
  (dispenses dispenser65 ingredient38)
  (dispenses dispenser292 ingredient59)
  (dispenses dispenser391 ingredient328)
  (clean shaker368)
  (clean shot280)
  (clean shot231)
  (empty shaker368)
  (empty shot280)
  (empty shot231)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker368 l0)
  (shaker-level shaker368 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail439 ingredient328)
  (cocktail-part2 cocktail439 ingredient59)
)
 (:goal
  (and
      (contains shot280 cocktail439)
)))
