(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot405 shot120 - shot
      ingredient417 ingredient317 - ingredient
      cocktail267 - cocktail
      dispenser80 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot405)
  (ontable shot120)
  (dispenses dispenser80 ingredient417)
  (dispenses dispenser390 ingredient317)
  (clean shaker358)
  (clean shot405)
  (clean shot120)
  (empty shaker358)
  (empty shot405)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail267 ingredient317)
  (cocktail-part2 cocktail267 ingredient417)
)
 (:goal
  (and
      (contains shot405 cocktail267)
)))
