(define (problem prob)
 (:domain barman)
 (:objects 
      shaker351 - shaker
      left right - hand
      shot33 shot17 - shot
      ingredient6 ingredient209 ingredient95 ingredient151 - ingredient
      cocktail480 - cocktail
      dispenser251 dispenser116 dispenser88 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker351)
  (ontable shot33)
  (ontable shot17)
  (dispenses dispenser251 ingredient6)
  (dispenses dispenser116 ingredient209)
  (dispenses dispenser88 ingredient95)
  (dispenses dispenser449 ingredient151)
  (clean shaker351)
  (clean shot33)
  (clean shot17)
  (empty shaker351)
  (empty shot33)
  (empty shot17)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker351 l0)
  (shaker-level shaker351 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient209)
  (cocktail-part2 cocktail480 ingredient95)
)
 (:goal
  (and
      (contains shot33 cocktail480)
)))
