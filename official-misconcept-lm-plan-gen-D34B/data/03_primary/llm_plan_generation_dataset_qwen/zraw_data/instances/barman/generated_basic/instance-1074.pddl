(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot112 shot79 shot297 - shot
      ingredient348 ingredient142 ingredient483 ingredient484 - ingredient
      cocktail1 - cocktail
      dispenser407 dispenser62 dispenser370 dispenser357 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot112)
  (ontable shot79)
  (ontable shot297)
  (dispenses dispenser407 ingredient348)
  (dispenses dispenser62 ingredient142)
  (dispenses dispenser370 ingredient483)
  (dispenses dispenser357 ingredient484)
  (clean shaker357)
  (clean shot112)
  (clean shot79)
  (clean shot297)
  (empty shaker357)
  (empty shot112)
  (empty shot79)
  (empty shot297)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient484)
  (cocktail-part2 cocktail1 ingredient142)
)
 (:goal
  (and
      (contains shot112 cocktail1)
      (contains shot79 ingredient348)
)))
