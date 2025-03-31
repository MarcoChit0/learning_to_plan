(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot231 - shot
      ingredient454 ingredient179 - ingredient
      cocktail206 - cocktail
      dispenser387 dispenser270 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot231)
  (dispenses dispenser387 ingredient454)
  (dispenses dispenser270 ingredient179)
  (clean shaker241)
  (clean shot231)
  (empty shaker241)
  (empty shot231)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient179)
  (cocktail-part2 cocktail206 ingredient454)
)
 (:goal
  (and
      (contains shot231 cocktail206)
)))
