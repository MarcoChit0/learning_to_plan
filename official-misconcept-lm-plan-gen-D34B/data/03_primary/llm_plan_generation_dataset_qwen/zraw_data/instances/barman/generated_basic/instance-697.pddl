(define (problem prob)
 (:domain barman)
 (:objects 
      shaker147 - shaker
      left right - hand
      shot255 shot314 - shot
      ingredient1 ingredient499 ingredient132 - ingredient
      cocktail1 - cocktail
      dispenser212 dispenser445 dispenser23 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker147)
  (ontable shot255)
  (ontable shot314)
  (dispenses dispenser212 ingredient1)
  (dispenses dispenser445 ingredient499)
  (dispenses dispenser23 ingredient132)
  (clean shaker147)
  (clean shot255)
  (clean shot314)
  (empty shaker147)
  (empty shot255)
  (empty shot314)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker147 l0)
  (shaker-level shaker147 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient1)
  (cocktail-part2 cocktail1 ingredient132)
)
 (:goal
  (and
      (contains shot255 cocktail1)
)))
