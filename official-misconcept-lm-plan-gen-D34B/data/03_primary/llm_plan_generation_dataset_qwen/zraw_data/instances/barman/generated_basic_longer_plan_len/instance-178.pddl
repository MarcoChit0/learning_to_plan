(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot378 shot319 shot448 shot90 - shot
      ingredient320 ingredient171 - ingredient
      cocktail1 - cocktail
      dispenser235 dispenser54 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot378)
  (ontable shot319)
  (ontable shot448)
  (ontable shot90)
  (dispenses dispenser235 ingredient320)
  (dispenses dispenser54 ingredient171)
  (clean shaker103)
  (clean shot378)
  (clean shot319)
  (clean shot448)
  (clean shot90)
  (empty shaker103)
  (empty shot378)
  (empty shot319)
  (empty shot448)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient320)
  (cocktail-part2 cocktail1 ingredient171)
)
 (:goal
  (and
      (contains shot378 cocktail1)
      (contains shot319 ingredient320)
      (contains shot448 ingredient320)
)))
