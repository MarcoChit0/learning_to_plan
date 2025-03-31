(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot475 - shot
      ingredient336 ingredient143 ingredient390 - ingredient
      cocktail498 - cocktail
      dispenser232 dispenser33 dispenser227 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot475)
  (dispenses dispenser232 ingredient336)
  (dispenses dispenser33 ingredient143)
  (dispenses dispenser227 ingredient390)
  (clean shaker19)
  (clean shot475)
  (empty shaker19)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail498 ingredient336)
  (cocktail-part2 cocktail498 ingredient143)
)
 (:goal
  (and
      (contains shot475 cocktail498)
)))
