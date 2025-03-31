(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot274 shot394 - shot
      ingredient202 ingredient237 - ingredient
      cocktail291 - cocktail
      dispenser459 dispenser84 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot274)
  (ontable shot394)
  (dispenses dispenser459 ingredient202)
  (dispenses dispenser84 ingredient237)
  (clean shaker491)
  (clean shot274)
  (clean shot394)
  (empty shaker491)
  (empty shot274)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail291 ingredient237)
  (cocktail-part2 cocktail291 ingredient202)
)
 (:goal
  (and
      (contains shot274 cocktail291)
)))
