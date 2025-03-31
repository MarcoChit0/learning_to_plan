(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot236 - shot
      ingredient69 ingredient398 ingredient270 - ingredient
      cocktail1 - cocktail
      dispenser201 dispenser285 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot236)
  (dispenses dispenser201 ingredient69)
  (dispenses dispenser285 ingredient398)
  (dispenses dispenser310 ingredient270)
  (clean shaker486)
  (clean shot236)
  (empty shaker486)
  (empty shot236)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient69)
  (cocktail-part2 cocktail1 ingredient398)
)
 (:goal
  (and
      (contains shot236 cocktail1)
)))
