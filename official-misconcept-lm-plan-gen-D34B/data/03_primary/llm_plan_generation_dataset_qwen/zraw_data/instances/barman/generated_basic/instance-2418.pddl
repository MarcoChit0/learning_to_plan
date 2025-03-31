(define (problem prob)
 (:domain barman)
 (:objects 
      shaker93 - shaker
      left right - hand
      shot85 - shot
      ingredient349 ingredient187 - ingredient
      cocktail335 - cocktail
      dispenser365 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker93)
  (ontable shot85)
  (dispenses dispenser365 ingredient349)
  (dispenses dispenser292 ingredient187)
  (clean shaker93)
  (clean shot85)
  (empty shaker93)
  (empty shot85)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker93 l0)
  (shaker-level shaker93 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail335 ingredient349)
  (cocktail-part2 cocktail335 ingredient187)
)
 (:goal
  (and
      (contains shot85 cocktail335)
)))
