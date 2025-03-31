(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot306 shot221 - shot
      ingredient442 ingredient34 ingredient284 ingredient224 - ingredient
      cocktail465 - cocktail
      dispenser253 dispenser496 dispenser238 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot306)
  (ontable shot221)
  (dispenses dispenser253 ingredient442)
  (dispenses dispenser496 ingredient34)
  (dispenses dispenser238 ingredient284)
  (dispenses dispenser263 ingredient224)
  (clean shaker99)
  (clean shot306)
  (clean shot221)
  (empty shaker99)
  (empty shot306)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail465 ingredient284)
  (cocktail-part2 cocktail465 ingredient224)
)
 (:goal
  (and
      (contains shot306 cocktail465)
)))
