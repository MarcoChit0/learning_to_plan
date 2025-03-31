(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot61 - shot
      ingredient43 ingredient212 - ingredient
      cocktail1 - cocktail
      dispenser359 dispenser261 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot61)
  (dispenses dispenser359 ingredient43)
  (dispenses dispenser261 ingredient212)
  (clean shaker237)
  (clean shot61)
  (empty shaker237)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient43)
  (cocktail-part2 cocktail1 ingredient212)
)
 (:goal
  (and
      (contains shot61 cocktail1)
)))
