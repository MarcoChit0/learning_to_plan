(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot185 - shot
      ingredient175 ingredient132 ingredient496 ingredient277 - ingredient
      cocktail41 - cocktail
      dispenser104 dispenser198 dispenser396 dispenser153 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot185)
  (dispenses dispenser104 ingredient175)
  (dispenses dispenser198 ingredient132)
  (dispenses dispenser396 ingredient496)
  (dispenses dispenser153 ingredient277)
  (clean shaker364)
  (clean shot185)
  (empty shaker364)
  (empty shot185)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail41 ingredient175)
  (cocktail-part2 cocktail41 ingredient132)
)
 (:goal
  (and
      (contains shot185 cocktail41)
)))
