(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot482 shot426 - shot
      ingredient349 ingredient138 ingredient217 ingredient449 - ingredient
      cocktail328 - cocktail
      dispenser68 dispenser15 dispenser374 dispenser309 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot482)
  (ontable shot426)
  (dispenses dispenser68 ingredient349)
  (dispenses dispenser15 ingredient138)
  (dispenses dispenser374 ingredient217)
  (dispenses dispenser309 ingredient449)
  (clean shaker55)
  (clean shot482)
  (clean shot426)
  (empty shaker55)
  (empty shot482)
  (empty shot426)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient449)
  (cocktail-part2 cocktail328 ingredient217)
)
 (:goal
  (and
      (contains shot482 cocktail328)
)))
