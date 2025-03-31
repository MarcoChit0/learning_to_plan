(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot42 shot448 - shot
      ingredient368 ingredient134 - ingredient
      cocktail197 - cocktail
      dispenser288 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot42)
  (ontable shot448)
  (dispenses dispenser288 ingredient368)
  (dispenses dispenser115 ingredient134)
  (clean shaker469)
  (clean shot42)
  (clean shot448)
  (empty shaker469)
  (empty shot42)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail197 ingredient368)
  (cocktail-part2 cocktail197 ingredient134)
)
 (:goal
  (and
      (contains shot42 cocktail197)
)))
