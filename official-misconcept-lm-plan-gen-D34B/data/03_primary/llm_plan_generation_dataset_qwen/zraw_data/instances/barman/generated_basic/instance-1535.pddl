(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot200 shot488 - shot
      ingredient293 ingredient471 ingredient17 ingredient397 - ingredient
      cocktail251 - cocktail
      dispenser93 dispenser155 dispenser65 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot200)
  (ontable shot488)
  (dispenses dispenser93 ingredient293)
  (dispenses dispenser155 ingredient471)
  (dispenses dispenser65 ingredient17)
  (dispenses dispenser166 ingredient397)
  (clean shaker156)
  (clean shot200)
  (clean shot488)
  (empty shaker156)
  (empty shot200)
  (empty shot488)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient471)
  (cocktail-part2 cocktail251 ingredient293)
)
 (:goal
  (and
      (contains shot200 cocktail251)
)))
