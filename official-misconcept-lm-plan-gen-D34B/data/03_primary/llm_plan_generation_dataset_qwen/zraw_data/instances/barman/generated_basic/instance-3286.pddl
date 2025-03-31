(define (problem prob)
 (:domain barman)
 (:objects 
      shaker81 - shaker
      left right - hand
      shot85 shot432 - shot
      ingredient284 ingredient150 ingredient301 - ingredient
      cocktail347 - cocktail
      dispenser437 dispenser167 dispenser233 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker81)
  (ontable shot85)
  (ontable shot432)
  (dispenses dispenser437 ingredient284)
  (dispenses dispenser167 ingredient150)
  (dispenses dispenser233 ingredient301)
  (clean shaker81)
  (clean shot85)
  (clean shot432)
  (empty shaker81)
  (empty shot85)
  (empty shot432)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker81 l0)
  (shaker-level shaker81 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail347 ingredient301)
  (cocktail-part2 cocktail347 ingredient150)
)
 (:goal
  (and
      (contains shot85 cocktail347)
)))
