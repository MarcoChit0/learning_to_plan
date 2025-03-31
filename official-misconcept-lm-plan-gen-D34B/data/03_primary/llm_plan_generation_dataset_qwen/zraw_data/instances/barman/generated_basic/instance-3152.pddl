(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot211 shot397 shot277 - shot
      ingredient295 ingredient182 ingredient386 ingredient185 - ingredient
      cocktail124 - cocktail
      dispenser292 dispenser450 dispenser22 dispenser476 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot211)
  (ontable shot397)
  (ontable shot277)
  (dispenses dispenser292 ingredient295)
  (dispenses dispenser450 ingredient182)
  (dispenses dispenser22 ingredient386)
  (dispenses dispenser476 ingredient185)
  (clean shaker212)
  (clean shot211)
  (clean shot397)
  (clean shot277)
  (empty shaker212)
  (empty shot211)
  (empty shot397)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient185)
  (cocktail-part2 cocktail124 ingredient182)
)
 (:goal
  (and
      (contains shot211 cocktail124)
      (contains shot397 cocktail124)
)))
