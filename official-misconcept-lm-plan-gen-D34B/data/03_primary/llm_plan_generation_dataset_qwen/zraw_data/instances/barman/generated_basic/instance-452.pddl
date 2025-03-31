(define (problem prob)
 (:domain barman)
 (:objects 
      shaker423 - shaker
      left right - hand
      shot443 shot371 - shot
      ingredient94 ingredient437 ingredient407 - ingredient
      cocktail1 - cocktail
      dispenser288 dispenser227 dispenser473 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker423)
  (ontable shot443)
  (ontable shot371)
  (dispenses dispenser288 ingredient94)
  (dispenses dispenser227 ingredient437)
  (dispenses dispenser473 ingredient407)
  (clean shaker423)
  (clean shot443)
  (clean shot371)
  (empty shaker423)
  (empty shot443)
  (empty shot371)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker423 l0)
  (shaker-level shaker423 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient407)
  (cocktail-part2 cocktail1 ingredient437)
)
 (:goal
  (and
      (contains shot443 cocktail1)
)))
