(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot421 - shot
      ingredient477 ingredient448 ingredient153 - ingredient
      cocktail1 - cocktail
      dispenser192 dispenser243 dispenser210 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot421)
  (dispenses dispenser192 ingredient477)
  (dispenses dispenser243 ingredient448)
  (dispenses dispenser210 ingredient153)
  (clean shaker67)
  (clean shot421)
  (empty shaker67)
  (empty shot421)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient477)
  (cocktail-part2 cocktail1 ingredient448)
)
 (:goal
  (and
      (contains shot421 cocktail1)
)))
