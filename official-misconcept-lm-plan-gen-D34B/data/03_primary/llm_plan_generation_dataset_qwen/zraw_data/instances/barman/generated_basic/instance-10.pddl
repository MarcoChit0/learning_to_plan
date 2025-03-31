(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot183 shot71 - shot
      ingredient288 ingredient17 - ingredient
      cocktail1 - cocktail
      dispenser274 dispenser248 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot183)
  (ontable shot71)
  (dispenses dispenser274 ingredient288)
  (dispenses dispenser248 ingredient17)
  (clean shaker364)
  (clean shot183)
  (clean shot71)
  (empty shaker364)
  (empty shot183)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient288)
  (cocktail-part2 cocktail1 ingredient17)
)
 (:goal
  (and
      (contains shot183 cocktail1)
)))
