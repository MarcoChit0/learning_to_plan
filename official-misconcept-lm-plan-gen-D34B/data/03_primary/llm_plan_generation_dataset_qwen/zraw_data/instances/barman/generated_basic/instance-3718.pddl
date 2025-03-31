(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot1 - shot
      ingredient78 ingredient176 ingredient125 ingredient346 - ingredient
      cocktail137 - cocktail
      dispenser34 dispenser277 dispenser265 dispenser280 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot1)
  (dispenses dispenser34 ingredient78)
  (dispenses dispenser277 ingredient176)
  (dispenses dispenser265 ingredient125)
  (dispenses dispenser280 ingredient346)
  (clean shaker162)
  (clean shot1)
  (empty shaker162)
  (empty shot1)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail137 ingredient176)
  (cocktail-part2 cocktail137 ingredient78)
)
 (:goal
  (and
      (contains shot1 cocktail137)
)))
