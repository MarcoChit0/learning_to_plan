(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot163 shot412 shot332 - shot
      ingredient304 ingredient246 ingredient450 - ingredient
      cocktail140 - cocktail
      dispenser404 dispenser156 dispenser261 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot163)
  (ontable shot412)
  (ontable shot332)
  (dispenses dispenser404 ingredient304)
  (dispenses dispenser156 ingredient246)
  (dispenses dispenser261 ingredient450)
  (clean shaker372)
  (clean shot163)
  (clean shot412)
  (clean shot332)
  (empty shaker372)
  (empty shot163)
  (empty shot412)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient450)
  (cocktail-part2 cocktail140 ingredient246)
)
 (:goal
  (and
      (contains shot163 cocktail140)
      (contains shot412 cocktail140)
)))
