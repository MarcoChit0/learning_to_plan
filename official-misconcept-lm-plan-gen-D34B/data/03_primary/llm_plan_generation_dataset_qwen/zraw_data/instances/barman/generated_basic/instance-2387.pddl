(define (problem prob)
 (:domain barman)
 (:objects 
      shaker9 - shaker
      left right - hand
      shot229 shot427 - shot
      ingredient431 ingredient69 ingredient348 - ingredient
      cocktail473 - cocktail
      dispenser398 dispenser138 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker9)
  (ontable shot229)
  (ontable shot427)
  (dispenses dispenser398 ingredient431)
  (dispenses dispenser138 ingredient69)
  (dispenses dispenser16 ingredient348)
  (clean shaker9)
  (clean shot229)
  (clean shot427)
  (empty shaker9)
  (empty shot229)
  (empty shot427)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker9 l0)
  (shaker-level shaker9 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient69)
  (cocktail-part2 cocktail473 ingredient348)
)
 (:goal
  (and
      (contains shot229 cocktail473)
)))
