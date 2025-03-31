(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot463 shot57 shot433 - shot
      ingredient404 ingredient247 - ingredient
      cocktail279 - cocktail
      dispenser307 dispenser55 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot463)
  (ontable shot57)
  (ontable shot433)
  (dispenses dispenser307 ingredient404)
  (dispenses dispenser55 ingredient247)
  (clean shaker478)
  (clean shot463)
  (clean shot57)
  (clean shot433)
  (empty shaker478)
  (empty shot463)
  (empty shot57)
  (empty shot433)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail279 ingredient404)
  (cocktail-part2 cocktail279 ingredient247)
)
 (:goal
  (and
      (contains shot463 cocktail279)
      (contains shot57 cocktail279)
)))
