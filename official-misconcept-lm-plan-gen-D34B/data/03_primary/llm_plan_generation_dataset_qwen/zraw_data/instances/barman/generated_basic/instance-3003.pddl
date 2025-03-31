(define (problem prob)
 (:domain barman)
 (:objects 
      shaker285 - shaker
      left right - hand
      shot33 shot291 - shot
      ingredient153 ingredient288 ingredient3 - ingredient
      cocktail447 - cocktail
      dispenser208 dispenser453 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker285)
  (ontable shot33)
  (ontable shot291)
  (dispenses dispenser208 ingredient153)
  (dispenses dispenser453 ingredient288)
  (dispenses dispenser273 ingredient3)
  (clean shaker285)
  (clean shot33)
  (clean shot291)
  (empty shaker285)
  (empty shot33)
  (empty shot291)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker285 l0)
  (shaker-level shaker285 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail447 ingredient153)
  (cocktail-part2 cocktail447 ingredient288)
)
 (:goal
  (and
      (contains shot33 cocktail447)
)))
