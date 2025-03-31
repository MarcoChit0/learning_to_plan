(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot396 - shot
      ingredient290 ingredient428 - ingredient
      cocktail1 - cocktail
      dispenser143 dispenser327 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot396)
  (dispenses dispenser143 ingredient290)
  (dispenses dispenser327 ingredient428)
  (clean shaker352)
  (clean shot396)
  (empty shaker352)
  (empty shot396)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient290)
  (cocktail-part2 cocktail1 ingredient428)
)
 (:goal
  (and
      (contains shot396 cocktail1)
)))
