(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot271 shot233 - shot
      ingredient436 ingredient89 ingredient304 - ingredient
      cocktail106 - cocktail
      dispenser297 dispenser217 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot271)
  (ontable shot233)
  (dispenses dispenser297 ingredient436)
  (dispenses dispenser217 ingredient89)
  (dispenses dispenser313 ingredient304)
  (clean shaker426)
  (clean shot271)
  (clean shot233)
  (empty shaker426)
  (empty shot271)
  (empty shot233)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail106 ingredient304)
  (cocktail-part2 cocktail106 ingredient89)
)
 (:goal
  (and
      (contains shot271 cocktail106)
)))
