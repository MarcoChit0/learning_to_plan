(define (problem prob)
 (:domain barman)
 (:objects 
      shaker122 - shaker
      left right - hand
      shot275 shot188 - shot
      ingredient220 ingredient125 - ingredient
      cocktail32 - cocktail
      dispenser156 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker122)
  (ontable shot275)
  (ontable shot188)
  (dispenses dispenser156 ingredient220)
  (dispenses dispenser446 ingredient125)
  (clean shaker122)
  (clean shot275)
  (clean shot188)
  (empty shaker122)
  (empty shot275)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker122 l0)
  (shaker-level shaker122 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient220)
  (cocktail-part2 cocktail32 ingredient125)
)
 (:goal
  (and
      (contains shot275 cocktail32)
)))
