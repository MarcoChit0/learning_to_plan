(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot172 shot254 - shot
      ingredient129 ingredient226 ingredient406 ingredient105 - ingredient
      cocktail489 - cocktail
      dispenser451 dispenser454 dispenser406 dispenser179 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot172)
  (ontable shot254)
  (dispenses dispenser451 ingredient129)
  (dispenses dispenser454 ingredient226)
  (dispenses dispenser406 ingredient406)
  (dispenses dispenser179 ingredient105)
  (clean shaker498)
  (clean shot172)
  (clean shot254)
  (empty shaker498)
  (empty shot172)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail489 ingredient129)
  (cocktail-part2 cocktail489 ingredient406)
)
 (:goal
  (and
      (contains shot172 cocktail489)
)))
