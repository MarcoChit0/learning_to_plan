(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot18 - shot
      ingredient270 ingredient390 ingredient441 - ingredient
      cocktail1 - cocktail
      dispenser113 dispenser483 dispenser129 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot18)
  (dispenses dispenser113 ingredient270)
  (dispenses dispenser483 ingredient390)
  (dispenses dispenser129 ingredient441)
  (clean shaker463)
  (clean shot18)
  (empty shaker463)
  (empty shot18)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient441)
  (cocktail-part2 cocktail1 ingredient270)
)
 (:goal
  (and
      (contains shot18 cocktail1)
)))
