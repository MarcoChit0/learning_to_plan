(define (problem prob)
 (:domain barman)
 (:objects 
      shaker494 - shaker
      left right - hand
      shot155 - shot
      ingredient150 ingredient317 - ingredient
      cocktail355 - cocktail
      dispenser404 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker494)
  (ontable shot155)
  (dispenses dispenser404 ingredient150)
  (dispenses dispenser286 ingredient317)
  (clean shaker494)
  (clean shot155)
  (empty shaker494)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker494 l0)
  (shaker-level shaker494 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient317)
  (cocktail-part2 cocktail355 ingredient150)
)
 (:goal
  (and
      (contains shot155 cocktail355)
)))
