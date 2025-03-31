(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot391 - shot
      ingredient94 ingredient392 - ingredient
      cocktail461 - cocktail
      dispenser495 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot391)
  (dispenses dispenser495 ingredient94)
  (dispenses dispenser5 ingredient392)
  (clean shaker26)
  (clean shot391)
  (empty shaker26)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient94)
  (cocktail-part2 cocktail461 ingredient392)
)
 (:goal
  (and
      (contains shot391 cocktail461)
)))
