(define (problem prob)
 (:domain barman)
 (:objects 
      shaker140 - shaker
      left right - hand
      shot442 - shot
      ingredient395 ingredient457 ingredient112 ingredient273 - ingredient
      cocktail482 - cocktail
      dispenser204 dispenser285 dispenser104 dispenser353 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker140)
  (ontable shot442)
  (dispenses dispenser204 ingredient395)
  (dispenses dispenser285 ingredient457)
  (dispenses dispenser104 ingredient112)
  (dispenses dispenser353 ingredient273)
  (clean shaker140)
  (clean shot442)
  (empty shaker140)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker140 l0)
  (shaker-level shaker140 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient273)
  (cocktail-part2 cocktail482 ingredient112)
)
 (:goal
  (and
      (contains shot442 cocktail482)
)))
