(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot291 shot137 - shot
      ingredient86 ingredient193 - ingredient
      cocktail1 - cocktail
      dispenser79 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot291)
  (ontable shot137)
  (dispenses dispenser79 ingredient86)
  (dispenses dispenser114 ingredient193)
  (clean shaker388)
  (clean shot291)
  (clean shot137)
  (empty shaker388)
  (empty shot291)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient193)
  (cocktail-part2 cocktail1 ingredient86)
)
 (:goal
  (and
      (contains shot291 cocktail1)
)))
