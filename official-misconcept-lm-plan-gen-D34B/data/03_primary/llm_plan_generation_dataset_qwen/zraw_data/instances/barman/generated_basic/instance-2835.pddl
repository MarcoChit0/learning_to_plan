(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot430 shot65 - shot
      ingredient481 ingredient449 ingredient430 ingredient395 - ingredient
      cocktail388 - cocktail
      dispenser228 dispenser465 dispenser235 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot430)
  (ontable shot65)
  (dispenses dispenser228 ingredient481)
  (dispenses dispenser465 ingredient449)
  (dispenses dispenser235 ingredient430)
  (dispenses dispenser92 ingredient395)
  (clean shaker437)
  (clean shot430)
  (clean shot65)
  (empty shaker437)
  (empty shot430)
  (empty shot65)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient481)
  (cocktail-part2 cocktail388 ingredient395)
)
 (:goal
  (and
      (contains shot430 cocktail388)
)))
