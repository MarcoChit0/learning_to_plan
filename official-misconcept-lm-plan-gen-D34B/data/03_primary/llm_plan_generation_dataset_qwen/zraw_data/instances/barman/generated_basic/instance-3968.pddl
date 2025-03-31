(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot331 shot487 - shot
      ingredient405 ingredient326 ingredient181 ingredient186 - ingredient
      cocktail215 - cocktail
      dispenser167 dispenser490 dispenser376 dispenser294 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot331)
  (ontable shot487)
  (dispenses dispenser167 ingredient405)
  (dispenses dispenser490 ingredient326)
  (dispenses dispenser376 ingredient181)
  (dispenses dispenser294 ingredient186)
  (clean shaker62)
  (clean shot331)
  (clean shot487)
  (empty shaker62)
  (empty shot331)
  (empty shot487)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail215 ingredient405)
  (cocktail-part2 cocktail215 ingredient326)
)
 (:goal
  (and
      (contains shot331 cocktail215)
)))
