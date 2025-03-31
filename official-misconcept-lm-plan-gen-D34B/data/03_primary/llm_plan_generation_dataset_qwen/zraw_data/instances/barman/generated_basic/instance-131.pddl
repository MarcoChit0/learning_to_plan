(define (problem prob)
 (:domain barman)
 (:objects 
      shaker5 - shaker
      left right - hand
      shot41 shot453 - shot
      ingredient254 ingredient303 - ingredient
      cocktail1 - cocktail
      dispenser53 dispenser237 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker5)
  (ontable shot41)
  (ontable shot453)
  (dispenses dispenser53 ingredient254)
  (dispenses dispenser237 ingredient303)
  (clean shaker5)
  (clean shot41)
  (clean shot453)
  (empty shaker5)
  (empty shot41)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker5 l0)
  (shaker-level shaker5 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient303)
  (cocktail-part2 cocktail1 ingredient254)
)
 (:goal
  (and
      (contains shot41 cocktail1)
)))
