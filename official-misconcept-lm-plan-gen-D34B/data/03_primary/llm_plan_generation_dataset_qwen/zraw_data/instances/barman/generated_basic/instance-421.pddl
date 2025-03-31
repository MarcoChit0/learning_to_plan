(define (problem prob)
 (:domain barman)
 (:objects 
      shaker300 - shaker
      left right - hand
      shot114 - shot
      ingredient108 ingredient26 - ingredient
      cocktail1 - cocktail
      dispenser177 dispenser185 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker300)
  (ontable shot114)
  (dispenses dispenser177 ingredient108)
  (dispenses dispenser185 ingredient26)
  (clean shaker300)
  (clean shot114)
  (empty shaker300)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker300 l0)
  (shaker-level shaker300 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient26)
  (cocktail-part2 cocktail1 ingredient108)
)
 (:goal
  (and
      (contains shot114 cocktail1)
)))
