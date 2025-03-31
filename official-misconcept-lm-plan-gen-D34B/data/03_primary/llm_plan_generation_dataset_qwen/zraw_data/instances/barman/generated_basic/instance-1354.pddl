(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot111 - shot
      ingredient14 ingredient450 ingredient108 - ingredient
      cocktail285 - cocktail
      dispenser478 dispenser341 dispenser113 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot111)
  (dispenses dispenser478 ingredient14)
  (dispenses dispenser341 ingredient450)
  (dispenses dispenser113 ingredient108)
  (clean shaker426)
  (clean shot111)
  (empty shaker426)
  (empty shot111)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail285 ingredient450)
  (cocktail-part2 cocktail285 ingredient14)
)
 (:goal
  (and
      (contains shot111 cocktail285)
)))
