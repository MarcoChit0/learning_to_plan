(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot447 shot206 shot387 - shot
      ingredient27 ingredient303 ingredient221 ingredient450 - ingredient
      cocktail390 - cocktail
      dispenser261 dispenser53 dispenser168 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot447)
  (ontable shot206)
  (ontable shot387)
  (dispenses dispenser261 ingredient27)
  (dispenses dispenser53 ingredient303)
  (dispenses dispenser168 ingredient221)
  (dispenses dispenser92 ingredient450)
  (clean shaker405)
  (clean shot447)
  (clean shot206)
  (clean shot387)
  (empty shaker405)
  (empty shot447)
  (empty shot206)
  (empty shot387)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient27)
  (cocktail-part2 cocktail390 ingredient450)
)
 (:goal
  (and
      (contains shot447 cocktail390)
      (contains shot206 cocktail390)
)))
