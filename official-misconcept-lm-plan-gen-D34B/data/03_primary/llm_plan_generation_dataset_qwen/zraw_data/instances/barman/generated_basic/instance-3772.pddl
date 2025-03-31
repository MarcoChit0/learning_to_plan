(define (problem prob)
 (:domain barman)
 (:objects 
      shaker283 - shaker
      left right - hand
      shot310 shot446 shot243 - shot
      ingredient147 ingredient93 - ingredient
      cocktail102 - cocktail
      dispenser117 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker283)
  (ontable shot310)
  (ontable shot446)
  (ontable shot243)
  (dispenses dispenser117 ingredient147)
  (dispenses dispenser143 ingredient93)
  (clean shaker283)
  (clean shot310)
  (clean shot446)
  (clean shot243)
  (empty shaker283)
  (empty shot310)
  (empty shot446)
  (empty shot243)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker283 l0)
  (shaker-level shaker283 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail102 ingredient93)
  (cocktail-part2 cocktail102 ingredient147)
)
 (:goal
  (and
      (contains shot310 cocktail102)
      (contains shot446 cocktail102)
)))
