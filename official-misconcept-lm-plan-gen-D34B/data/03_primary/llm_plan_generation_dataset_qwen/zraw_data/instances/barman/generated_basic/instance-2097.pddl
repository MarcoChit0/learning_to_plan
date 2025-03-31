(define (problem prob)
 (:domain barman)
 (:objects 
      shaker2 - shaker
      left right - hand
      shot374 shot22 - shot
      ingredient461 ingredient252 - ingredient
      cocktail106 - cocktail
      dispenser85 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker2)
  (ontable shot374)
  (ontable shot22)
  (dispenses dispenser85 ingredient461)
  (dispenses dispenser292 ingredient252)
  (clean shaker2)
  (clean shot374)
  (clean shot22)
  (empty shaker2)
  (empty shot374)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker2 l0)
  (shaker-level shaker2 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail106 ingredient461)
  (cocktail-part2 cocktail106 ingredient252)
)
 (:goal
  (and
      (contains shot374 cocktail106)
)))
