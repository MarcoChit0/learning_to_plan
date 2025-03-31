(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot126 shot130 - shot
      ingredient281 ingredient347 ingredient70 ingredient486 - ingredient
      cocktail1 - cocktail
      dispenser378 dispenser218 dispenser360 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot126)
  (ontable shot130)
  (dispenses dispenser378 ingredient281)
  (dispenses dispenser218 ingredient347)
  (dispenses dispenser360 ingredient70)
  (dispenses dispenser30 ingredient486)
  (clean shaker270)
  (clean shot126)
  (clean shot130)
  (empty shaker270)
  (empty shot126)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient281)
  (cocktail-part2 cocktail1 ingredient70)
)
 (:goal
  (and
      (contains shot126 cocktail1)
)))
