(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot425 shot31 shot397 - shot
      ingredient197 ingredient19 - ingredient
      cocktail151 - cocktail
      dispenser68 dispenser298 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot425)
  (ontable shot31)
  (ontable shot397)
  (dispenses dispenser68 ingredient197)
  (dispenses dispenser298 ingredient19)
  (clean shaker15)
  (clean shot425)
  (clean shot31)
  (clean shot397)
  (empty shaker15)
  (empty shot425)
  (empty shot31)
  (empty shot397)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail151 ingredient197)
  (cocktail-part2 cocktail151 ingredient19)
)
 (:goal
  (and
      (contains shot425 cocktail151)
      (contains shot31 ingredient197)
)))
