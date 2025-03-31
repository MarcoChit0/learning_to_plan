(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot275 shot144 - shot
      ingredient309 ingredient262 ingredient324 - ingredient
      cocktail416 - cocktail
      dispenser279 dispenser434 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot275)
  (ontable shot144)
  (dispenses dispenser279 ingredient309)
  (dispenses dispenser434 ingredient262)
  (dispenses dispenser194 ingredient324)
  (clean shaker212)
  (clean shot275)
  (clean shot144)
  (empty shaker212)
  (empty shot275)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail416 ingredient309)
  (cocktail-part2 cocktail416 ingredient324)
)
 (:goal
  (and
      (contains shot275 cocktail416)
)))
