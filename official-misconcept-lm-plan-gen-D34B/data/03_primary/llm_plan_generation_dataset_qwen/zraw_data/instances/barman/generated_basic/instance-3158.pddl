(define (problem prob)
 (:domain barman)
 (:objects 
      shaker84 - shaker
      left right - hand
      shot357 shot47 shot316 - shot
      ingredient56 ingredient102 ingredient127 ingredient310 - ingredient
      cocktail431 - cocktail
      dispenser395 dispenser65 dispenser293 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker84)
  (ontable shot357)
  (ontable shot47)
  (ontable shot316)
  (dispenses dispenser395 ingredient56)
  (dispenses dispenser65 ingredient102)
  (dispenses dispenser293 ingredient127)
  (dispenses dispenser213 ingredient310)
  (clean shaker84)
  (clean shot357)
  (clean shot47)
  (clean shot316)
  (empty shaker84)
  (empty shot357)
  (empty shot47)
  (empty shot316)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker84 l0)
  (shaker-level shaker84 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient102)
  (cocktail-part2 cocktail431 ingredient56)
)
 (:goal
  (and
      (contains shot357 cocktail431)
      (contains shot47 ingredient310)
)))
