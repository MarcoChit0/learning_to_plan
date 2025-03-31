(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot433 shot176 - shot
      ingredient93 ingredient85 ingredient54 - ingredient
      cocktail325 - cocktail
      dispenser183 dispenser401 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot433)
  (ontable shot176)
  (dispenses dispenser183 ingredient93)
  (dispenses dispenser401 ingredient85)
  (dispenses dispenser200 ingredient54)
  (clean shaker236)
  (clean shot433)
  (clean shot176)
  (empty shaker236)
  (empty shot433)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail325 ingredient93)
  (cocktail-part2 cocktail325 ingredient54)
)
 (:goal
  (and
      (contains shot433 cocktail325)
)))
