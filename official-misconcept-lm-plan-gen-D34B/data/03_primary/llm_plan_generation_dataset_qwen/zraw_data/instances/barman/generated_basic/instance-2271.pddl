(define (problem prob)
 (:domain barman)
 (:objects 
      shaker312 - shaker
      left right - hand
      shot160 shot351 - shot
      ingredient165 ingredient466 ingredient363 ingredient244 - ingredient
      cocktail356 - cocktail
      dispenser239 dispenser230 dispenser315 dispenser10 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker312)
  (ontable shot160)
  (ontable shot351)
  (dispenses dispenser239 ingredient165)
  (dispenses dispenser230 ingredient466)
  (dispenses dispenser315 ingredient363)
  (dispenses dispenser10 ingredient244)
  (clean shaker312)
  (clean shot160)
  (clean shot351)
  (empty shaker312)
  (empty shot160)
  (empty shot351)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker312 l0)
  (shaker-level shaker312 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient244)
  (cocktail-part2 cocktail356 ingredient363)
)
 (:goal
  (and
      (contains shot160 cocktail356)
)))
