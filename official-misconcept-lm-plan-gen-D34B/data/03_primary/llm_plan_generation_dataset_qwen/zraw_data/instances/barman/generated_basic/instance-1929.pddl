(define (problem prob)
 (:domain barman)
 (:objects 
      shaker323 - shaker
      left right - hand
      shot18 - shot
      ingredient399 ingredient191 - ingredient
      cocktail115 - cocktail
      dispenser466 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker323)
  (ontable shot18)
  (dispenses dispenser466 ingredient399)
  (dispenses dispenser290 ingredient191)
  (clean shaker323)
  (clean shot18)
  (empty shaker323)
  (empty shot18)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker323 l0)
  (shaker-level shaker323 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient191)
  (cocktail-part2 cocktail115 ingredient399)
)
 (:goal
  (and
      (contains shot18 cocktail115)
)))
