(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot228 - shot
      ingredient95 ingredient16 - ingredient
      cocktail13 - cocktail
      dispenser309 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot228)
  (dispenses dispenser309 ingredient95)
  (dispenses dispenser292 ingredient16)
  (clean shaker190)
  (clean shot228)
  (empty shaker190)
  (empty shot228)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient16)
  (cocktail-part2 cocktail13 ingredient95)
)
 (:goal
  (and
      (contains shot228 cocktail13)
)))
