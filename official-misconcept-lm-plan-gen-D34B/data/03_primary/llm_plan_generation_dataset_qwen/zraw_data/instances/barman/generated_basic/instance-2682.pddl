(define (problem prob)
 (:domain barman)
 (:objects 
      shaker52 - shaker
      left right - hand
      shot140 shot170 - shot
      ingredient30 ingredient402 ingredient227 - ingredient
      cocktail229 - cocktail
      dispenser199 dispenser271 dispenser174 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker52)
  (ontable shot140)
  (ontable shot170)
  (dispenses dispenser199 ingredient30)
  (dispenses dispenser271 ingredient402)
  (dispenses dispenser174 ingredient227)
  (clean shaker52)
  (clean shot140)
  (clean shot170)
  (empty shaker52)
  (empty shot140)
  (empty shot170)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker52 l0)
  (shaker-level shaker52 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail229 ingredient402)
  (cocktail-part2 cocktail229 ingredient227)
)
 (:goal
  (and
      (contains shot140 cocktail229)
)))
