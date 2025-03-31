(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot413 - shot
      ingredient145 ingredient352 ingredient161 ingredient162 - ingredient
      cocktail1 - cocktail
      dispenser470 dispenser328 dispenser314 dispenser95 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot413)
  (dispenses dispenser470 ingredient145)
  (dispenses dispenser328 ingredient352)
  (dispenses dispenser314 ingredient161)
  (dispenses dispenser95 ingredient162)
  (clean shaker276)
  (clean shot413)
  (empty shaker276)
  (empty shot413)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient352)
  (cocktail-part2 cocktail1 ingredient145)
)
 (:goal
  (and
      (contains shot413 cocktail1)
)))
