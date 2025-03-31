(define (problem prob)
 (:domain barman)
 (:objects 
      shaker387 - shaker
      left right - hand
      shot244 shot231 shot415 - shot
      ingredient190 ingredient424 ingredient12 ingredient431 - ingredient
      cocktail266 - cocktail
      dispenser154 dispenser191 dispenser10 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker387)
  (ontable shot244)
  (ontable shot231)
  (ontable shot415)
  (dispenses dispenser154 ingredient190)
  (dispenses dispenser191 ingredient424)
  (dispenses dispenser10 ingredient12)
  (dispenses dispenser276 ingredient431)
  (clean shaker387)
  (clean shot244)
  (clean shot231)
  (clean shot415)
  (empty shaker387)
  (empty shot244)
  (empty shot231)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker387 l0)
  (shaker-level shaker387 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail266 ingredient431)
  (cocktail-part2 cocktail266 ingredient12)
)
 (:goal
  (and
      (contains shot244 cocktail266)
      (contains shot231 cocktail266)
)))
