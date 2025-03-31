(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot113 - shot
      ingredient126 ingredient32 - ingredient
      cocktail23 - cocktail
      dispenser195 dispenser215 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot113)
  (dispenses dispenser195 ingredient126)
  (dispenses dispenser215 ingredient32)
  (clean shaker207)
  (clean shot113)
  (empty shaker207)
  (empty shot113)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail23 ingredient126)
  (cocktail-part2 cocktail23 ingredient32)
)
 (:goal
  (and
      (contains shot113 cocktail23)
)))
