(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot81 shot286 shot94 - shot
      ingredient351 ingredient338 - ingredient
      cocktail228 - cocktail
      dispenser4 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot81)
  (ontable shot286)
  (ontable shot94)
  (dispenses dispenser4 ingredient351)
  (dispenses dispenser145 ingredient338)
  (clean shaker68)
  (clean shot81)
  (clean shot286)
  (clean shot94)
  (empty shaker68)
  (empty shot81)
  (empty shot286)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient351)
  (cocktail-part2 cocktail228 ingredient338)
)
 (:goal
  (and
      (contains shot81 cocktail228)
      (contains shot286 ingredient351)
)))
