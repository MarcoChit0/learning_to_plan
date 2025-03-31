(define (problem prob)
 (:domain barman)
 (:objects 
      shaker23 - shaker
      left right - hand
      shot348 - shot
      ingredient177 ingredient240 - ingredient
      cocktail254 - cocktail
      dispenser393 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker23)
  (ontable shot348)
  (dispenses dispenser393 ingredient177)
  (dispenses dispenser257 ingredient240)
  (clean shaker23)
  (clean shot348)
  (empty shaker23)
  (empty shot348)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker23 l0)
  (shaker-level shaker23 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail254 ingredient177)
  (cocktail-part2 cocktail254 ingredient240)
)
 (:goal
  (and
      (contains shot348 cocktail254)
)))
