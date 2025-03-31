(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot42 shot40 shot254 - shot
      ingredient214 ingredient211 ingredient361 ingredient315 - ingredient
      cocktail407 - cocktail
      dispenser428 dispenser456 dispenser195 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot42)
  (ontable shot40)
  (ontable shot254)
  (dispenses dispenser428 ingredient214)
  (dispenses dispenser456 ingredient211)
  (dispenses dispenser195 ingredient361)
  (dispenses dispenser133 ingredient315)
  (clean shaker77)
  (clean shot42)
  (clean shot40)
  (clean shot254)
  (empty shaker77)
  (empty shot42)
  (empty shot40)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail407 ingredient315)
  (cocktail-part2 cocktail407 ingredient361)
)
 (:goal
  (and
      (contains shot42 cocktail407)
      (contains shot40 ingredient361)
)))
