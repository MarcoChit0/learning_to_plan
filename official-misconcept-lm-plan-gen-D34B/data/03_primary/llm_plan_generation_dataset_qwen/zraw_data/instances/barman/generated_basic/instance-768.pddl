(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot449 - shot
      ingredient216 ingredient392 - ingredient
      cocktail1 - cocktail
      dispenser444 dispenser480 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot449)
  (dispenses dispenser444 ingredient216)
  (dispenses dispenser480 ingredient392)
  (clean shaker72)
  (clean shot449)
  (empty shaker72)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient216)
  (cocktail-part2 cocktail1 ingredient392)
)
 (:goal
  (and
      (contains shot449 cocktail1)
)))
