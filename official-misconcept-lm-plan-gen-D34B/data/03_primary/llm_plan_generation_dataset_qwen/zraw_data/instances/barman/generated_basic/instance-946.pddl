(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot112 shot385 shot195 - shot
      ingredient400 ingredient273 ingredient25 ingredient333 - ingredient
      cocktail1 - cocktail
      dispenser238 dispenser13 dispenser77 dispenser107 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot112)
  (ontable shot385)
  (ontable shot195)
  (dispenses dispenser238 ingredient400)
  (dispenses dispenser13 ingredient273)
  (dispenses dispenser77 ingredient25)
  (dispenses dispenser107 ingredient333)
  (clean shaker304)
  (clean shot112)
  (clean shot385)
  (clean shot195)
  (empty shaker304)
  (empty shot112)
  (empty shot385)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient333)
  (cocktail-part2 cocktail1 ingredient273)
)
 (:goal
  (and
      (contains shot112 cocktail1)
      (contains shot385 cocktail1)
)))
