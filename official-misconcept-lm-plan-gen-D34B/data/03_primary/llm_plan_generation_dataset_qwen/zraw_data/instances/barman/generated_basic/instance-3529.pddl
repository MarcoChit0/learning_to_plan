(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot15 shot60 - shot
      ingredient282 ingredient235 ingredient157 ingredient402 - ingredient
      cocktail101 - cocktail
      dispenser71 dispenser39 dispenser17 dispenser328 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot15)
  (ontable shot60)
  (dispenses dispenser71 ingredient282)
  (dispenses dispenser39 ingredient235)
  (dispenses dispenser17 ingredient157)
  (dispenses dispenser328 ingredient402)
  (clean shaker295)
  (clean shot15)
  (clean shot60)
  (empty shaker295)
  (empty shot15)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient157)
  (cocktail-part2 cocktail101 ingredient235)
)
 (:goal
  (and
      (contains shot15 cocktail101)
)))
