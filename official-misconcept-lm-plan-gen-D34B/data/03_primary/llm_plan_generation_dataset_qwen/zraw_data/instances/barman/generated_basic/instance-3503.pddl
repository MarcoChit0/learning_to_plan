(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot298 shot48 - shot
      ingredient123 ingredient353 ingredient15 ingredient399 - ingredient
      cocktail415 - cocktail
      dispenser170 dispenser421 dispenser420 dispenser494 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot298)
  (ontable shot48)
  (dispenses dispenser170 ingredient123)
  (dispenses dispenser421 ingredient353)
  (dispenses dispenser420 ingredient15)
  (dispenses dispenser494 ingredient399)
  (clean shaker441)
  (clean shot298)
  (clean shot48)
  (empty shaker441)
  (empty shot298)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail415 ingredient15)
  (cocktail-part2 cocktail415 ingredient399)
)
 (:goal
  (and
      (contains shot298 cocktail415)
)))
