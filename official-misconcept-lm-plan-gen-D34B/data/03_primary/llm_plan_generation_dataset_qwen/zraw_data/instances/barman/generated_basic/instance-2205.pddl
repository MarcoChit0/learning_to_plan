(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot31 shot418 shot104 - shot
      ingredient334 ingredient40 ingredient449 ingredient69 - ingredient
      cocktail438 - cocktail
      dispenser190 dispenser254 dispenser78 dispenser234 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot31)
  (ontable shot418)
  (ontable shot104)
  (dispenses dispenser190 ingredient334)
  (dispenses dispenser254 ingredient40)
  (dispenses dispenser78 ingredient449)
  (dispenses dispenser234 ingredient69)
  (clean shaker13)
  (clean shot31)
  (clean shot418)
  (clean shot104)
  (empty shaker13)
  (empty shot31)
  (empty shot418)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail438 ingredient334)
  (cocktail-part2 cocktail438 ingredient69)
)
 (:goal
  (and
      (contains shot31 cocktail438)
      (contains shot418 ingredient40)
)))
