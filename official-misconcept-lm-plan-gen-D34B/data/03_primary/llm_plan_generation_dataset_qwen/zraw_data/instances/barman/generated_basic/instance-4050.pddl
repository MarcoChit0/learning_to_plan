(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot457 shot80 - shot
      ingredient138 ingredient47 ingredient449 ingredient73 - ingredient
      cocktail173 - cocktail
      dispenser134 dispenser45 dispenser205 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot457)
  (ontable shot80)
  (dispenses dispenser134 ingredient138)
  (dispenses dispenser45 ingredient47)
  (dispenses dispenser205 ingredient449)
  (dispenses dispenser133 ingredient73)
  (clean shaker51)
  (clean shot457)
  (clean shot80)
  (empty shaker51)
  (empty shot457)
  (empty shot80)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail173 ingredient449)
  (cocktail-part2 cocktail173 ingredient47)
)
 (:goal
  (and
      (contains shot457 cocktail173)
)))
