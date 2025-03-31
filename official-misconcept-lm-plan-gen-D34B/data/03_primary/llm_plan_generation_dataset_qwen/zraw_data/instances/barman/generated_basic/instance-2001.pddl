(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot405 shot183 shot390 - shot
      ingredient383 ingredient114 ingredient70 - ingredient
      cocktail167 - cocktail
      dispenser296 dispenser435 dispenser300 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot405)
  (ontable shot183)
  (ontable shot390)
  (dispenses dispenser296 ingredient383)
  (dispenses dispenser435 ingredient114)
  (dispenses dispenser300 ingredient70)
  (clean shaker1)
  (clean shot405)
  (clean shot183)
  (clean shot390)
  (empty shaker1)
  (empty shot405)
  (empty shot183)
  (empty shot390)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail167 ingredient383)
  (cocktail-part2 cocktail167 ingredient70)
)
 (:goal
  (and
      (contains shot405 cocktail167)
      (contains shot183 ingredient70)
)))
