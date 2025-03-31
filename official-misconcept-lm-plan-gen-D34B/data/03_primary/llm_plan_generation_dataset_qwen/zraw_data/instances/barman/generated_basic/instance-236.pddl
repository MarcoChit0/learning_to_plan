(define (problem prob)
 (:domain barman)
 (:objects 
      shaker74 - shaker
      left right - hand
      shot183 - shot
      ingredient410 ingredient73 - ingredient
      cocktail1 - cocktail
      dispenser201 dispenser438 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker74)
  (ontable shot183)
  (dispenses dispenser201 ingredient410)
  (dispenses dispenser438 ingredient73)
  (clean shaker74)
  (clean shot183)
  (empty shaker74)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker74 l0)
  (shaker-level shaker74 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient410)
  (cocktail-part2 cocktail1 ingredient73)
)
 (:goal
  (and
      (contains shot183 cocktail1)
)))
