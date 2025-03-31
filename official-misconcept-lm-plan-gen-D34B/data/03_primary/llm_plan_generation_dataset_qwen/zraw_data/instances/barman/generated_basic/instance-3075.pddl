(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot128 shot155 - shot
      ingredient48 ingredient128 - ingredient
      cocktail359 - cocktail
      dispenser471 dispenser447 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot128)
  (ontable shot155)
  (dispenses dispenser471 ingredient48)
  (dispenses dispenser447 ingredient128)
  (clean shaker171)
  (clean shot128)
  (clean shot155)
  (empty shaker171)
  (empty shot128)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail359 ingredient48)
  (cocktail-part2 cocktail359 ingredient128)
)
 (:goal
  (and
      (contains shot128 cocktail359)
)))
