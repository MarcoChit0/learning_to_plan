(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot343 - shot
      ingredient15 ingredient119 ingredient112 - ingredient
      cocktail496 - cocktail
      dispenser329 dispenser93 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot343)
  (dispenses dispenser329 ingredient15)
  (dispenses dispenser93 ingredient119)
  (dispenses dispenser102 ingredient112)
  (clean shaker486)
  (clean shot343)
  (empty shaker486)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail496 ingredient112)
  (cocktail-part2 cocktail496 ingredient119)
)
 (:goal
  (and
      (contains shot343 cocktail496)
)))
