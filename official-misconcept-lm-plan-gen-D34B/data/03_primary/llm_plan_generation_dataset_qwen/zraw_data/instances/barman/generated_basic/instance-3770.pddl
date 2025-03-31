(define (problem prob)
 (:domain barman)
 (:objects 
      shaker386 - shaker
      left right - hand
      shot196 shot172 shot214 - shot
      ingredient429 ingredient322 - ingredient
      cocktail312 - cocktail
      dispenser351 dispenser347 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker386)
  (ontable shot196)
  (ontable shot172)
  (ontable shot214)
  (dispenses dispenser351 ingredient429)
  (dispenses dispenser347 ingredient322)
  (clean shaker386)
  (clean shot196)
  (clean shot172)
  (clean shot214)
  (empty shaker386)
  (empty shot196)
  (empty shot172)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker386 l0)
  (shaker-level shaker386 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail312 ingredient429)
  (cocktail-part2 cocktail312 ingredient322)
)
 (:goal
  (and
      (contains shot196 cocktail312)
      (contains shot172 cocktail312)
)))
