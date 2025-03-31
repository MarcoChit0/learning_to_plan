(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot423 shot331 shot472 - shot
      ingredient449 ingredient2 ingredient386 - ingredient
      cocktail153 - cocktail
      dispenser57 dispenser58 dispenser229 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot423)
  (ontable shot331)
  (ontable shot472)
  (dispenses dispenser57 ingredient449)
  (dispenses dispenser58 ingredient2)
  (dispenses dispenser229 ingredient386)
  (clean shaker120)
  (clean shot423)
  (clean shot331)
  (clean shot472)
  (empty shaker120)
  (empty shot423)
  (empty shot331)
  (empty shot472)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail153 ingredient2)
  (cocktail-part2 cocktail153 ingredient386)
)
 (:goal
  (and
      (contains shot423 cocktail153)
      (contains shot331 cocktail153)
)))
