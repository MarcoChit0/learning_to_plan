(define (problem prob)
 (:domain barman)
 (:objects 
      shaker217 - shaker
      left right - hand
      shot462 shot243 shot281 - shot
      ingredient334 ingredient27 ingredient63 ingredient408 - ingredient
      cocktail74 - cocktail
      dispenser416 dispenser86 dispenser9 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker217)
  (ontable shot462)
  (ontable shot243)
  (ontable shot281)
  (dispenses dispenser416 ingredient334)
  (dispenses dispenser86 ingredient27)
  (dispenses dispenser9 ingredient63)
  (dispenses dispenser73 ingredient408)
  (clean shaker217)
  (clean shot462)
  (clean shot243)
  (clean shot281)
  (empty shaker217)
  (empty shot462)
  (empty shot243)
  (empty shot281)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker217 l0)
  (shaker-level shaker217 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient63)
  (cocktail-part2 cocktail74 ingredient408)
)
 (:goal
  (and
      (contains shot462 cocktail74)
      (contains shot243 cocktail74)
)))
