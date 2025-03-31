(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot28 shot286 shot395 - shot
      ingredient0 ingredient86 ingredient490 ingredient135 - ingredient
      cocktail67 - cocktail
      dispenser149 dispenser376 dispenser15 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot28)
  (ontable shot286)
  (ontable shot395)
  (dispenses dispenser149 ingredient0)
  (dispenses dispenser376 ingredient86)
  (dispenses dispenser15 ingredient490)
  (dispenses dispenser463 ingredient135)
  (clean shaker459)
  (clean shot28)
  (clean shot286)
  (clean shot395)
  (empty shaker459)
  (empty shot28)
  (empty shot286)
  (empty shot395)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail67 ingredient135)
  (cocktail-part2 cocktail67 ingredient490)
)
 (:goal
  (and
      (contains shot28 cocktail67)
      (contains shot286 cocktail67)
)))
