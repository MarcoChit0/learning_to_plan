(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot53 shot454 - shot
      ingredient122 ingredient413 - ingredient
      cocktail1 - cocktail
      dispenser190 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot53)
  (ontable shot454)
  (dispenses dispenser190 ingredient122)
  (dispenses dispenser463 ingredient413)
  (clean shaker18)
  (clean shot53)
  (clean shot454)
  (empty shaker18)
  (empty shot53)
  (empty shot454)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient122)
  (cocktail-part2 cocktail1 ingredient413)
)
 (:goal
  (and
      (contains shot53 cocktail1)
)))
