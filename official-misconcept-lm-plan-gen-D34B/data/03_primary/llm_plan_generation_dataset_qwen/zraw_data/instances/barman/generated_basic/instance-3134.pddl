(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot81 shot340 shot133 - shot
      ingredient139 ingredient343 - ingredient
      cocktail17 - cocktail
      dispenser192 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot81)
  (ontable shot340)
  (ontable shot133)
  (dispenses dispenser192 ingredient139)
  (dispenses dispenser477 ingredient343)
  (clean shaker436)
  (clean shot81)
  (clean shot340)
  (clean shot133)
  (empty shaker436)
  (empty shot81)
  (empty shot340)
  (empty shot133)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail17 ingredient343)
  (cocktail-part2 cocktail17 ingredient139)
)
 (:goal
  (and
      (contains shot81 cocktail17)
      (contains shot340 ingredient343)
)))
