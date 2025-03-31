(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot210 shot373 - shot
      ingredient97 ingredient270 ingredient155 ingredient152 - ingredient
      cocktail170 - cocktail
      dispenser414 dispenser440 dispenser112 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot210)
  (ontable shot373)
  (dispenses dispenser414 ingredient97)
  (dispenses dispenser440 ingredient270)
  (dispenses dispenser112 ingredient155)
  (dispenses dispenser428 ingredient152)
  (clean shaker357)
  (clean shot210)
  (clean shot373)
  (empty shaker357)
  (empty shot210)
  (empty shot373)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail170 ingredient270)
  (cocktail-part2 cocktail170 ingredient155)
)
 (:goal
  (and
      (contains shot210 cocktail170)
)))
