(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot19 shot69 - shot
      ingredient270 ingredient469 ingredient196 ingredient493 - ingredient
      cocktail351 - cocktail
      dispenser164 dispenser282 dispenser258 dispenser481 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot19)
  (ontable shot69)
  (dispenses dispenser164 ingredient270)
  (dispenses dispenser282 ingredient469)
  (dispenses dispenser258 ingredient196)
  (dispenses dispenser481 ingredient493)
  (clean shaker380)
  (clean shot19)
  (clean shot69)
  (empty shaker380)
  (empty shot19)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail351 ingredient493)
  (cocktail-part2 cocktail351 ingredient196)
)
 (:goal
  (and
      (contains shot19 cocktail351)
)))
