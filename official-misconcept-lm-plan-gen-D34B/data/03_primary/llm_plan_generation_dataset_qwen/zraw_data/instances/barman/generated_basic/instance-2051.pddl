(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot254 - shot
      ingredient397 ingredient408 ingredient329 ingredient46 - ingredient
      cocktail177 - cocktail
      dispenser453 dispenser204 dispenser49 dispenser469 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot254)
  (dispenses dispenser453 ingredient397)
  (dispenses dispenser204 ingredient408)
  (dispenses dispenser49 ingredient329)
  (dispenses dispenser469 ingredient46)
  (clean shaker373)
  (clean shot254)
  (empty shaker373)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail177 ingredient408)
  (cocktail-part2 cocktail177 ingredient397)
)
 (:goal
  (and
      (contains shot254 cocktail177)
)))
