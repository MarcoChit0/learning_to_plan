(define (problem prob)
 (:domain barman)
 (:objects 
      shaker153 - shaker
      left right - hand
      shot462 shot117 - shot
      ingredient490 ingredient1 - ingredient
      cocktail305 - cocktail
      dispenser277 dispenser67 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker153)
  (ontable shot462)
  (ontable shot117)
  (dispenses dispenser277 ingredient490)
  (dispenses dispenser67 ingredient1)
  (clean shaker153)
  (clean shot462)
  (clean shot117)
  (empty shaker153)
  (empty shot462)
  (empty shot117)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker153 l0)
  (shaker-level shaker153 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail305 ingredient490)
  (cocktail-part2 cocktail305 ingredient1)
)
 (:goal
  (and
      (contains shot462 cocktail305)
)))
