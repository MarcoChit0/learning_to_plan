(define (problem prob)
 (:domain barman)
 (:objects 
      shaker82 - shaker
      left right - hand
      shot310 shot216 - shot
      ingredient173 ingredient494 ingredient460 - ingredient
      cocktail1 - cocktail
      dispenser301 dispenser131 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker82)
  (ontable shot310)
  (ontable shot216)
  (dispenses dispenser301 ingredient173)
  (dispenses dispenser131 ingredient494)
  (dispenses dispenser154 ingredient460)
  (clean shaker82)
  (clean shot310)
  (clean shot216)
  (empty shaker82)
  (empty shot310)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker82 l0)
  (shaker-level shaker82 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient460)
  (cocktail-part2 cocktail1 ingredient494)
)
 (:goal
  (and
      (contains shot310 cocktail1)
)))
