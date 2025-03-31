(define (problem prob)
 (:domain barman)
 (:objects 
      shaker402 - shaker
      left right - hand
      shot389 shot16 - shot
      ingredient497 ingredient144 ingredient5 ingredient307 - ingredient
      cocktail387 - cocktail
      dispenser465 dispenser96 dispenser81 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker402)
  (ontable shot389)
  (ontable shot16)
  (dispenses dispenser465 ingredient497)
  (dispenses dispenser96 ingredient144)
  (dispenses dispenser81 ingredient5)
  (dispenses dispenser128 ingredient307)
  (clean shaker402)
  (clean shot389)
  (clean shot16)
  (empty shaker402)
  (empty shot389)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker402 l0)
  (shaker-level shaker402 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail387 ingredient497)
  (cocktail-part2 cocktail387 ingredient5)
)
 (:goal
  (and
      (contains shot389 cocktail387)
)))
