(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot140 - shot
      ingredient188 ingredient277 ingredient95 - ingredient
      cocktail84 - cocktail
      dispenser24 dispenser86 dispenser198 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot140)
  (dispenses dispenser24 ingredient188)
  (dispenses dispenser86 ingredient277)
  (dispenses dispenser198 ingredient95)
  (clean shaker187)
  (clean shot140)
  (empty shaker187)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail84 ingredient188)
  (cocktail-part2 cocktail84 ingredient277)
)
 (:goal
  (and
      (contains shot140 cocktail84)
)))
