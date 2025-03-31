(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot163 shot446 - shot
      ingredient475 ingredient31 ingredient154 - ingredient
      cocktail87 - cocktail
      dispenser472 dispenser496 dispenser14 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot163)
  (ontable shot446)
  (dispenses dispenser472 ingredient475)
  (dispenses dispenser496 ingredient31)
  (dispenses dispenser14 ingredient154)
  (clean shaker119)
  (clean shot163)
  (clean shot446)
  (empty shaker119)
  (empty shot163)
  (empty shot446)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail87 ingredient31)
  (cocktail-part2 cocktail87 ingredient154)
)
 (:goal
  (and
      (contains shot163 cocktail87)
)))
