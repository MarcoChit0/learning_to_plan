(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot348 shot499 - shot
      ingredient73 ingredient492 - ingredient
      cocktail290 - cocktail
      dispenser148 dispenser156 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot348)
  (ontable shot499)
  (dispenses dispenser148 ingredient73)
  (dispenses dispenser156 ingredient492)
  (clean shaker236)
  (clean shot348)
  (clean shot499)
  (empty shaker236)
  (empty shot348)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail290 ingredient73)
  (cocktail-part2 cocktail290 ingredient492)
)
 (:goal
  (and
      (contains shot348 cocktail290)
)))
