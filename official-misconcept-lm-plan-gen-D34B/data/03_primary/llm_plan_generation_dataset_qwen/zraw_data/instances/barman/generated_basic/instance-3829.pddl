(define (problem prob)
 (:domain barman)
 (:objects 
      shaker195 - shaker
      left right - hand
      shot123 shot241 - shot
      ingredient391 ingredient79 ingredient413 - ingredient
      cocktail390 - cocktail
      dispenser456 dispenser234 dispenser81 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker195)
  (ontable shot123)
  (ontable shot241)
  (dispenses dispenser456 ingredient391)
  (dispenses dispenser234 ingredient79)
  (dispenses dispenser81 ingredient413)
  (clean shaker195)
  (clean shot123)
  (clean shot241)
  (empty shaker195)
  (empty shot123)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker195 l0)
  (shaker-level shaker195 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient413)
  (cocktail-part2 cocktail390 ingredient79)
)
 (:goal
  (and
      (contains shot123 cocktail390)
)))
