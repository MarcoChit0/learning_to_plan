(define (problem prob)
 (:domain barman)
 (:objects 
      shaker227 - shaker
      left right - hand
      shot401 - shot
      ingredient489 ingredient133 ingredient198 ingredient70 - ingredient
      cocktail49 - cocktail
      dispenser297 dispenser169 dispenser52 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker227)
  (ontable shot401)
  (dispenses dispenser297 ingredient489)
  (dispenses dispenser169 ingredient133)
  (dispenses dispenser52 ingredient198)
  (dispenses dispenser466 ingredient70)
  (clean shaker227)
  (clean shot401)
  (empty shaker227)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker227 l0)
  (shaker-level shaker227 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail49 ingredient489)
  (cocktail-part2 cocktail49 ingredient133)
)
 (:goal
  (and
      (contains shot401 cocktail49)
)))
