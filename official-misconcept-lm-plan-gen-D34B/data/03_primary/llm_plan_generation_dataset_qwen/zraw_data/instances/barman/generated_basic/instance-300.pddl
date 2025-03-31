(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot377 shot250 - shot
      ingredient147 ingredient424 - ingredient
      cocktail1 - cocktail
      dispenser132 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot377)
  (ontable shot250)
  (dispenses dispenser132 ingredient147)
  (dispenses dispenser222 ingredient424)
  (clean shaker497)
  (clean shot377)
  (clean shot250)
  (empty shaker497)
  (empty shot377)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient424)
  (cocktail-part2 cocktail1 ingredient147)
)
 (:goal
  (and
      (contains shot377 cocktail1)
)))
