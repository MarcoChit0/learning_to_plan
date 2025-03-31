(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot179 shot178 - shot
      ingredient411 ingredient415 ingredient392 - ingredient
      cocktail342 - cocktail
      dispenser200 dispenser451 dispenser31 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot179)
  (ontable shot178)
  (dispenses dispenser200 ingredient411)
  (dispenses dispenser451 ingredient415)
  (dispenses dispenser31 ingredient392)
  (clean shaker303)
  (clean shot179)
  (clean shot178)
  (empty shaker303)
  (empty shot179)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail342 ingredient392)
  (cocktail-part2 cocktail342 ingredient411)
)
 (:goal
  (and
      (contains shot179 cocktail342)
)))
