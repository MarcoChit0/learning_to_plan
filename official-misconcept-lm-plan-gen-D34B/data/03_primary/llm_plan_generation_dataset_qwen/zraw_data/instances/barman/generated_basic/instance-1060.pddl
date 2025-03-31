(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot154 shot387 shot213 - shot
      ingredient231 ingredient486 ingredient399 ingredient63 - ingredient
      cocktail1 - cocktail
      dispenser19 dispenser412 dispenser486 dispenser51 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot154)
  (ontable shot387)
  (ontable shot213)
  (dispenses dispenser19 ingredient231)
  (dispenses dispenser412 ingredient486)
  (dispenses dispenser486 ingredient399)
  (dispenses dispenser51 ingredient63)
  (clean shaker170)
  (clean shot154)
  (clean shot387)
  (clean shot213)
  (empty shaker170)
  (empty shot154)
  (empty shot387)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient486)
  (cocktail-part2 cocktail1 ingredient231)
)
 (:goal
  (and
      (contains shot154 cocktail1)
      (contains shot387 cocktail1)
)))
