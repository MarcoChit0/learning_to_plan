(define (problem prob)
 (:domain barman)
 (:objects 
      shaker434 - shaker
      left right - hand
      shot190 - shot
      ingredient481 ingredient375 ingredient437 ingredient304 - ingredient
      cocktail171 - cocktail
      dispenser348 dispenser104 dispenser211 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker434)
  (ontable shot190)
  (dispenses dispenser348 ingredient481)
  (dispenses dispenser104 ingredient375)
  (dispenses dispenser211 ingredient437)
  (dispenses dispenser271 ingredient304)
  (clean shaker434)
  (clean shot190)
  (empty shaker434)
  (empty shot190)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker434 l0)
  (shaker-level shaker434 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail171 ingredient304)
  (cocktail-part2 cocktail171 ingredient437)
)
 (:goal
  (and
      (contains shot190 cocktail171)
)))
