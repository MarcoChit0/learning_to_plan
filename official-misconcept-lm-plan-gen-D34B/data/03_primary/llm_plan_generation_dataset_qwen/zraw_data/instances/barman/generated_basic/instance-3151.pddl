(define (problem prob)
 (:domain barman)
 (:objects 
      shaker263 - shaker
      left right - hand
      shot385 shot491 shot216 - shot
      ingredient267 ingredient441 ingredient499 ingredient485 - ingredient
      cocktail475 - cocktail
      dispenser499 dispenser192 dispenser344 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker263)
  (ontable shot385)
  (ontable shot491)
  (ontable shot216)
  (dispenses dispenser499 ingredient267)
  (dispenses dispenser192 ingredient441)
  (dispenses dispenser344 ingredient499)
  (dispenses dispenser250 ingredient485)
  (clean shaker263)
  (clean shot385)
  (clean shot491)
  (clean shot216)
  (empty shaker263)
  (empty shot385)
  (empty shot491)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker263 l0)
  (shaker-level shaker263 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient485)
  (cocktail-part2 cocktail475 ingredient441)
)
 (:goal
  (and
      (contains shot385 cocktail475)
      (contains shot491 ingredient485)
)))
