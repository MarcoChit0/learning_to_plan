(define (problem prob)
 (:domain barman)
 (:objects 
      shaker254 - shaker
      left right - hand
      shot311 - shot
      ingredient163 ingredient321 - ingredient
      cocktail1 - cocktail
      dispenser297 dispenser481 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker254)
  (ontable shot311)
  (dispenses dispenser297 ingredient163)
  (dispenses dispenser481 ingredient321)
  (clean shaker254)
  (clean shot311)
  (empty shaker254)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker254 l0)
  (shaker-level shaker254 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient163)
  (cocktail-part2 cocktail1 ingredient321)
)
 (:goal
  (and
      (contains shot311 cocktail1)
)))
