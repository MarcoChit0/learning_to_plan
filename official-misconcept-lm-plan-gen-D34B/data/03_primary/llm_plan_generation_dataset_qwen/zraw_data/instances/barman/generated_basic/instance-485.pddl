(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot30 shot64 - shot
      ingredient143 ingredient94 ingredient304 ingredient248 - ingredient
      cocktail1 - cocktail
      dispenser302 dispenser497 dispenser315 dispenser91 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot30)
  (ontable shot64)
  (dispenses dispenser302 ingredient143)
  (dispenses dispenser497 ingredient94)
  (dispenses dispenser315 ingredient304)
  (dispenses dispenser91 ingredient248)
  (clean shaker375)
  (clean shot30)
  (clean shot64)
  (empty shaker375)
  (empty shot30)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient304)
  (cocktail-part2 cocktail1 ingredient94)
)
 (:goal
  (and
      (contains shot30 cocktail1)
)))
