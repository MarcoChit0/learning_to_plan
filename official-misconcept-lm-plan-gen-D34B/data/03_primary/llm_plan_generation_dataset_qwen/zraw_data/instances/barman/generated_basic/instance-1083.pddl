(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot57 - shot
      ingredient462 ingredient310 ingredient131 - ingredient
      cocktail1 - cocktail
      dispenser279 dispenser306 dispenser261 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot57)
  (dispenses dispenser279 ingredient462)
  (dispenses dispenser306 ingredient310)
  (dispenses dispenser261 ingredient131)
  (clean shaker181)
  (clean shot57)
  (empty shaker181)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient310)
  (cocktail-part2 cocktail1 ingredient131)
)
 (:goal
  (and
      (contains shot57 cocktail1)
)))
