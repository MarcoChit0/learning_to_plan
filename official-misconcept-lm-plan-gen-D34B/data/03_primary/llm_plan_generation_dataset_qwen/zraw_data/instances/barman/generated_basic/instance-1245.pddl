(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot106 shot7 shot479 - shot
      ingredient183 ingredient190 ingredient268 ingredient445 - ingredient
      cocktail295 - cocktail
      dispenser67 dispenser154 dispenser488 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot106)
  (ontable shot7)
  (ontable shot479)
  (dispenses dispenser67 ingredient183)
  (dispenses dispenser154 ingredient190)
  (dispenses dispenser488 ingredient268)
  (dispenses dispenser374 ingredient445)
  (clean shaker301)
  (clean shot106)
  (clean shot7)
  (clean shot479)
  (empty shaker301)
  (empty shot106)
  (empty shot7)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail295 ingredient268)
  (cocktail-part2 cocktail295 ingredient190)
)
 (:goal
  (and
      (contains shot106 cocktail295)
      (contains shot7 ingredient183)
)))
