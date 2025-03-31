(define (problem prob)
 (:domain barman)
 (:objects 
      shaker129 - shaker
      left right - hand
      shot198 shot51 - shot
      ingredient331 ingredient190 ingredient79 ingredient76 - ingredient
      cocktail50 - cocktail
      dispenser24 dispenser493 dispenser453 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker129)
  (ontable shot198)
  (ontable shot51)
  (dispenses dispenser24 ingredient331)
  (dispenses dispenser493 ingredient190)
  (dispenses dispenser453 ingredient79)
  (dispenses dispenser240 ingredient76)
  (clean shaker129)
  (clean shot198)
  (clean shot51)
  (empty shaker129)
  (empty shot198)
  (empty shot51)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker129 l0)
  (shaker-level shaker129 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail50 ingredient331)
  (cocktail-part2 cocktail50 ingredient79)
)
 (:goal
  (and
      (contains shot198 cocktail50)
)))
