(define (problem prob)
 (:domain barman)
 (:objects 
      shaker300 - shaker
      left right - hand
      shot248 - shot
      ingredient297 ingredient330 - ingredient
      cocktail1 - cocktail
      dispenser444 dispenser81 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker300)
  (ontable shot248)
  (dispenses dispenser444 ingredient297)
  (dispenses dispenser81 ingredient330)
  (clean shaker300)
  (clean shot248)
  (empty shaker300)
  (empty shot248)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker300 l0)
  (shaker-level shaker300 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient297)
  (cocktail-part2 cocktail1 ingredient330)
)
 (:goal
  (and
      (contains shot248 cocktail1)
)))
