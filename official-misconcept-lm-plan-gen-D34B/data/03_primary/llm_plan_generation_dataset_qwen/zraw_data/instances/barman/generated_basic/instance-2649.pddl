(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot226 shot340 - shot
      ingredient0 ingredient342 ingredient258 - ingredient
      cocktail210 - cocktail
      dispenser14 dispenser10 dispenser24 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot226)
  (ontable shot340)
  (dispenses dispenser14 ingredient0)
  (dispenses dispenser10 ingredient342)
  (dispenses dispenser24 ingredient258)
  (clean shaker15)
  (clean shot226)
  (clean shot340)
  (empty shaker15)
  (empty shot226)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail210 ingredient258)
  (cocktail-part2 cocktail210 ingredient342)
)
 (:goal
  (and
      (contains shot226 cocktail210)
)))
