(define (problem prob)
 (:domain barman)
 (:objects 
      shaker199 - shaker
      left right - hand
      shot88 shot155 - shot
      ingredient130 ingredient477 ingredient258 ingredient304 - ingredient
      cocktail37 - cocktail
      dispenser255 dispenser204 dispenser489 dispenser389 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker199)
  (ontable shot88)
  (ontable shot155)
  (dispenses dispenser255 ingredient130)
  (dispenses dispenser204 ingredient477)
  (dispenses dispenser489 ingredient258)
  (dispenses dispenser389 ingredient304)
  (clean shaker199)
  (clean shot88)
  (clean shot155)
  (empty shaker199)
  (empty shot88)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker199 l0)
  (shaker-level shaker199 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail37 ingredient130)
  (cocktail-part2 cocktail37 ingredient477)
)
 (:goal
  (and
      (contains shot88 cocktail37)
)))
