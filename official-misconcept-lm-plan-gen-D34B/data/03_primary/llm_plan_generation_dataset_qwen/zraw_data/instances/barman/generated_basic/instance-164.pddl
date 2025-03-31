(define (problem prob)
 (:domain barman)
 (:objects 
      shaker252 - shaker
      left right - hand
      shot272 shot476 - shot
      ingredient337 ingredient319 ingredient258 - ingredient
      cocktail1 - cocktail
      dispenser112 dispenser179 dispenser234 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker252)
  (ontable shot272)
  (ontable shot476)
  (dispenses dispenser112 ingredient337)
  (dispenses dispenser179 ingredient319)
  (dispenses dispenser234 ingredient258)
  (clean shaker252)
  (clean shot272)
  (clean shot476)
  (empty shaker252)
  (empty shot272)
  (empty shot476)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker252 l0)
  (shaker-level shaker252 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient258)
  (cocktail-part2 cocktail1 ingredient337)
)
 (:goal
  (and
      (contains shot272 cocktail1)
)))
