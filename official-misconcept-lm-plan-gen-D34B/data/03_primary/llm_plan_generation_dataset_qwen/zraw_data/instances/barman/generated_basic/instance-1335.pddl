(define (problem prob)
 (:domain barman)
 (:objects 
      shaker154 - shaker
      left right - hand
      shot10 shot114 shot354 - shot
      ingredient262 ingredient430 ingredient247 - ingredient
      cocktail475 - cocktail
      dispenser322 dispenser265 dispenser436 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker154)
  (ontable shot10)
  (ontable shot114)
  (ontable shot354)
  (dispenses dispenser322 ingredient262)
  (dispenses dispenser265 ingredient430)
  (dispenses dispenser436 ingredient247)
  (clean shaker154)
  (clean shot10)
  (clean shot114)
  (clean shot354)
  (empty shaker154)
  (empty shot10)
  (empty shot114)
  (empty shot354)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker154 l0)
  (shaker-level shaker154 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient430)
  (cocktail-part2 cocktail475 ingredient247)
)
 (:goal
  (and
      (contains shot10 cocktail475)
      (contains shot114 cocktail475)
)))
