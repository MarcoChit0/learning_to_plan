(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot356 shot294 - shot
      ingredient401 ingredient306 ingredient257 ingredient408 - ingredient
      cocktail361 - cocktail
      dispenser253 dispenser294 dispenser282 dispenser56 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot356)
  (ontable shot294)
  (dispenses dispenser253 ingredient401)
  (dispenses dispenser294 ingredient306)
  (dispenses dispenser282 ingredient257)
  (dispenses dispenser56 ingredient408)
  (clean shaker380)
  (clean shot356)
  (clean shot294)
  (empty shaker380)
  (empty shot356)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail361 ingredient408)
  (cocktail-part2 cocktail361 ingredient401)
)
 (:goal
  (and
      (contains shot356 cocktail361)
)))
