(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot347 - shot
      ingredient92 ingredient421 ingredient59 ingredient451 - ingredient
      cocktail260 - cocktail
      dispenser93 dispenser200 dispenser48 dispenser497 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot347)
  (dispenses dispenser93 ingredient92)
  (dispenses dispenser200 ingredient421)
  (dispenses dispenser48 ingredient59)
  (dispenses dispenser497 ingredient451)
  (clean shaker446)
  (clean shot347)
  (empty shaker446)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient451)
  (cocktail-part2 cocktail260 ingredient421)
)
 (:goal
  (and
      (contains shot347 cocktail260)
)))
