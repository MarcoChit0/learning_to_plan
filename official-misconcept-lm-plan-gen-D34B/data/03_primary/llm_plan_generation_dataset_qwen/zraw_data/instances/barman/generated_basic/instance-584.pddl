(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot492 shot282 - shot
      ingredient109 ingredient24 - ingredient
      cocktail1 - cocktail
      dispenser220 dispenser496 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot492)
  (ontable shot282)
  (dispenses dispenser220 ingredient109)
  (dispenses dispenser496 ingredient24)
  (clean shaker13)
  (clean shot492)
  (clean shot282)
  (empty shaker13)
  (empty shot492)
  (empty shot282)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient109)
  (cocktail-part2 cocktail1 ingredient24)
)
 (:goal
  (and
      (contains shot492 cocktail1)
)))
