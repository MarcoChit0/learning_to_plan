(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot254 shot428 shot317 - shot
      ingredient45 ingredient194 - ingredient
      cocktail481 - cocktail
      dispenser400 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot254)
  (ontable shot428)
  (ontable shot317)
  (dispenses dispenser400 ingredient45)
  (dispenses dispenser243 ingredient194)
  (clean shaker468)
  (clean shot254)
  (clean shot428)
  (clean shot317)
  (empty shaker468)
  (empty shot254)
  (empty shot428)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail481 ingredient45)
  (cocktail-part2 cocktail481 ingredient194)
)
 (:goal
  (and
      (contains shot254 cocktail481)
      (contains shot428 cocktail481)
)))
