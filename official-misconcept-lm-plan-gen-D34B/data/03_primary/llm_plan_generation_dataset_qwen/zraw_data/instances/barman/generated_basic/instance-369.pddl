(define (problem prob)
 (:domain barman)
 (:objects 
      shaker422 - shaker
      left right - hand
      shot347 - shot
      ingredient81 ingredient74 - ingredient
      cocktail1 - cocktail
      dispenser469 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker422)
  (ontable shot347)
  (dispenses dispenser469 ingredient81)
  (dispenses dispenser257 ingredient74)
  (clean shaker422)
  (clean shot347)
  (empty shaker422)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker422 l0)
  (shaker-level shaker422 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient81)
  (cocktail-part2 cocktail1 ingredient74)
)
 (:goal
  (and
      (contains shot347 cocktail1)
)))
