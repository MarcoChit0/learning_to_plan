(define (problem prob)
 (:domain barman)
 (:objects 
      shaker105 - shaker
      left right - hand
      shot228 shot180 - shot
      ingredient178 ingredient218 - ingredient
      cocktail1 - cocktail
      dispenser167 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker105)
  (ontable shot228)
  (ontable shot180)
  (dispenses dispenser167 ingredient178)
  (dispenses dispenser313 ingredient218)
  (clean shaker105)
  (clean shot228)
  (clean shot180)
  (empty shaker105)
  (empty shot228)
  (empty shot180)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker105 l0)
  (shaker-level shaker105 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient178)
  (cocktail-part2 cocktail1 ingredient218)
)
 (:goal
  (and
      (contains shot228 cocktail1)
)))
