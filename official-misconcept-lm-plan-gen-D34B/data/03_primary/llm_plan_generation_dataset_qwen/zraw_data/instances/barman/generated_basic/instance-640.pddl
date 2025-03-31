(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot93 shot489 shot214 - shot
      ingredient162 ingredient1 ingredient38 - ingredient
      cocktail1 - cocktail
      dispenser125 dispenser369 dispenser7 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot93)
  (ontable shot489)
  (ontable shot214)
  (dispenses dispenser125 ingredient162)
  (dispenses dispenser369 ingredient1)
  (dispenses dispenser7 ingredient38)
  (clean shaker89)
  (clean shot93)
  (clean shot489)
  (clean shot214)
  (empty shaker89)
  (empty shot93)
  (empty shot489)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient162)
  (cocktail-part2 cocktail1 ingredient38)
)
 (:goal
  (and
      (contains shot93 cocktail1)
      (contains shot489 cocktail1)
)))
