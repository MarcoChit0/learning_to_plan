(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot20 - shot
      ingredient146 ingredient447 ingredient304 ingredient425 - ingredient
      cocktail440 - cocktail
      dispenser24 dispenser72 dispenser358 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot20)
  (dispenses dispenser24 ingredient146)
  (dispenses dispenser72 ingredient447)
  (dispenses dispenser358 ingredient304)
  (dispenses dispenser61 ingredient425)
  (clean shaker48)
  (clean shot20)
  (empty shaker48)
  (empty shot20)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail440 ingredient425)
  (cocktail-part2 cocktail440 ingredient304)
)
 (:goal
  (and
      (contains shot20 cocktail440)
)))
