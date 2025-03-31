(define (problem prob)
 (:domain barman)
 (:objects 
      shaker483 - shaker
      left right - hand
      shot122 shot199 - shot
      ingredient218 ingredient175 ingredient376 ingredient156 - ingredient
      cocktail288 - cocktail
      dispenser92 dispenser15 dispenser343 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker483)
  (ontable shot122)
  (ontable shot199)
  (dispenses dispenser92 ingredient218)
  (dispenses dispenser15 ingredient175)
  (dispenses dispenser343 ingredient376)
  (dispenses dispenser69 ingredient156)
  (clean shaker483)
  (clean shot122)
  (clean shot199)
  (empty shaker483)
  (empty shot122)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker483 l0)
  (shaker-level shaker483 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail288 ingredient376)
  (cocktail-part2 cocktail288 ingredient156)
)
 (:goal
  (and
      (contains shot122 cocktail288)
)))
