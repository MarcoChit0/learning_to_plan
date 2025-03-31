(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot180 shot332 - shot
      ingredient464 ingredient279 ingredient159 ingredient263 - ingredient
      cocktail131 - cocktail
      dispenser235 dispenser13 dispenser60 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot180)
  (ontable shot332)
  (dispenses dispenser235 ingredient464)
  (dispenses dispenser13 ingredient279)
  (dispenses dispenser60 ingredient159)
  (dispenses dispenser418 ingredient263)
  (clean shaker486)
  (clean shot180)
  (clean shot332)
  (empty shaker486)
  (empty shot180)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail131 ingredient159)
  (cocktail-part2 cocktail131 ingredient263)
)
 (:goal
  (and
      (contains shot180 cocktail131)
)))
