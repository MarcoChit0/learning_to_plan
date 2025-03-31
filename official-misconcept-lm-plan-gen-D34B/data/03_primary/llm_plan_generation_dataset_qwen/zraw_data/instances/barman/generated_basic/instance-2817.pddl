(define (problem prob)
 (:domain barman)
 (:objects 
      shaker123 - shaker
      left right - hand
      shot201 shot198 shot308 - shot
      ingredient310 ingredient464 ingredient474 ingredient191 - ingredient
      cocktail280 - cocktail
      dispenser20 dispenser214 dispenser351 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker123)
  (ontable shot201)
  (ontable shot198)
  (ontable shot308)
  (dispenses dispenser20 ingredient310)
  (dispenses dispenser214 ingredient464)
  (dispenses dispenser351 ingredient474)
  (dispenses dispenser478 ingredient191)
  (clean shaker123)
  (clean shot201)
  (clean shot198)
  (clean shot308)
  (empty shaker123)
  (empty shot201)
  (empty shot198)
  (empty shot308)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker123 l0)
  (shaker-level shaker123 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail280 ingredient474)
  (cocktail-part2 cocktail280 ingredient191)
)
 (:goal
  (and
      (contains shot201 cocktail280)
      (contains shot198 cocktail280)
)))
