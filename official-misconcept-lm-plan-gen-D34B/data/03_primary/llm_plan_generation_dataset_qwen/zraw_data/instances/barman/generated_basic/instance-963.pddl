(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot349 - shot
      ingredient360 ingredient231 ingredient437 ingredient363 - ingredient
      cocktail1 - cocktail
      dispenser18 dispenser206 dispenser475 dispenser245 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot349)
  (dispenses dispenser18 ingredient360)
  (dispenses dispenser206 ingredient231)
  (dispenses dispenser475 ingredient437)
  (dispenses dispenser245 ingredient363)
  (clean shaker189)
  (clean shot349)
  (empty shaker189)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient437)
  (cocktail-part2 cocktail1 ingredient360)
)
 (:goal
  (and
      (contains shot349 cocktail1)
)))
