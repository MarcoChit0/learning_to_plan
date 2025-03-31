(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot388 shot448 - shot
      ingredient386 ingredient166 ingredient237 ingredient476 - ingredient
      cocktail254 - cocktail
      dispenser192 dispenser22 dispenser374 dispenser304 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot388)
  (ontable shot448)
  (dispenses dispenser192 ingredient386)
  (dispenses dispenser22 ingredient166)
  (dispenses dispenser374 ingredient237)
  (dispenses dispenser304 ingredient476)
  (clean shaker467)
  (clean shot388)
  (clean shot448)
  (empty shaker467)
  (empty shot388)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail254 ingredient386)
  (cocktail-part2 cocktail254 ingredient166)
)
 (:goal
  (and
      (contains shot388 cocktail254)
)))
