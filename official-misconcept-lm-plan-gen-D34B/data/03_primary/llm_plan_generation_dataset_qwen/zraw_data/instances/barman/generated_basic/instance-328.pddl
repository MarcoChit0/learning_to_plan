(define (problem prob)
 (:domain barman)
 (:objects 
      shaker52 - shaker
      left right - hand
      shot5 - shot
      ingredient5 ingredient26 - ingredient
      cocktail1 - cocktail
      dispenser128 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker52)
  (ontable shot5)
  (dispenses dispenser128 ingredient5)
  (dispenses dispenser196 ingredient26)
  (clean shaker52)
  (clean shot5)
  (empty shaker52)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker52 l0)
  (shaker-level shaker52 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient26)
  (cocktail-part2 cocktail1 ingredient5)
)
 (:goal
  (and
      (contains shot5 cocktail1)
)))
