(define (problem prob)
 (:domain barman)
 (:objects 
      shaker309 - shaker
      left right - hand
      shot410 - shot
      ingredient230 ingredient200 - ingredient
      cocktail39 - cocktail
      dispenser134 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker309)
  (ontable shot410)
  (dispenses dispenser134 ingredient230)
  (dispenses dispenser477 ingredient200)
  (clean shaker309)
  (clean shot410)
  (empty shaker309)
  (empty shot410)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker309 l0)
  (shaker-level shaker309 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient230)
  (cocktail-part2 cocktail39 ingredient200)
)
 (:goal
  (and
      (contains shot410 cocktail39)
)))
