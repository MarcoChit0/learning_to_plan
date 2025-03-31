(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot78 shot354 shot123 - shot
      ingredient434 ingredient270 - ingredient
      cocktail39 - cocktail
      dispenser205 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot78)
  (ontable shot354)
  (ontable shot123)
  (dispenses dispenser205 ingredient434)
  (dispenses dispenser321 ingredient270)
  (clean shaker356)
  (clean shot78)
  (clean shot354)
  (clean shot123)
  (empty shaker356)
  (empty shot78)
  (empty shot354)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient434)
  (cocktail-part2 cocktail39 ingredient270)
)
 (:goal
  (and
      (contains shot78 cocktail39)
      (contains shot354 cocktail39)
)))
