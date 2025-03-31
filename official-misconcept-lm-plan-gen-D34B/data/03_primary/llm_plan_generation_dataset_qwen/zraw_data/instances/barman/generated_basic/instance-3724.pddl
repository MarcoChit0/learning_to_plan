(define (problem prob)
 (:domain barman)
 (:objects 
      shaker46 - shaker
      left right - hand
      shot125 - shot
      ingredient235 ingredient96 ingredient499 ingredient196 - ingredient
      cocktail352 - cocktail
      dispenser42 dispenser268 dispenser187 dispenser94 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker46)
  (ontable shot125)
  (dispenses dispenser42 ingredient235)
  (dispenses dispenser268 ingredient96)
  (dispenses dispenser187 ingredient499)
  (dispenses dispenser94 ingredient196)
  (clean shaker46)
  (clean shot125)
  (empty shaker46)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker46 l0)
  (shaker-level shaker46 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient196)
  (cocktail-part2 cocktail352 ingredient235)
)
 (:goal
  (and
      (contains shot125 cocktail352)
)))
