(define (problem prob)
 (:domain barman)
 (:objects 
      shaker9 - shaker
      left right - hand
      shot109 shot192 - shot
      ingredient32 ingredient346 ingredient282 ingredient148 - ingredient
      cocktail1 - cocktail
      dispenser369 dispenser115 dispenser484 dispenser371 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker9)
  (ontable shot109)
  (ontable shot192)
  (dispenses dispenser369 ingredient32)
  (dispenses dispenser115 ingredient346)
  (dispenses dispenser484 ingredient282)
  (dispenses dispenser371 ingredient148)
  (clean shaker9)
  (clean shot109)
  (clean shot192)
  (empty shaker9)
  (empty shot109)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker9 l0)
  (shaker-level shaker9 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient346)
  (cocktail-part2 cocktail1 ingredient282)
)
 (:goal
  (and
      (contains shot109 cocktail1)
)))
