(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot79 - shot
      ingredient151 ingredient142 - ingredient
      cocktail1 - cocktail
      dispenser210 dispenser382 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot79)
  (dispenses dispenser210 ingredient151)
  (dispenses dispenser382 ingredient142)
  (clean shaker467)
  (clean shot79)
  (empty shaker467)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient142)
  (cocktail-part2 cocktail1 ingredient151)
)
 (:goal
  (and
      (contains shot79 cocktail1)
)))
