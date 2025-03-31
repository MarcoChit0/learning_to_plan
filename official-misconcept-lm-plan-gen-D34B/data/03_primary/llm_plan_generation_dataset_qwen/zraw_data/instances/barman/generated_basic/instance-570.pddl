(define (problem prob)
 (:domain barman)
 (:objects 
      shaker106 - shaker
      left right - hand
      shot385 shot137 - shot
      ingredient142 ingredient224 ingredient214 - ingredient
      cocktail1 - cocktail
      dispenser111 dispenser456 dispenser211 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker106)
  (ontable shot385)
  (ontable shot137)
  (dispenses dispenser111 ingredient142)
  (dispenses dispenser456 ingredient224)
  (dispenses dispenser211 ingredient214)
  (clean shaker106)
  (clean shot385)
  (clean shot137)
  (empty shaker106)
  (empty shot385)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker106 l0)
  (shaker-level shaker106 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient214)
  (cocktail-part2 cocktail1 ingredient224)
)
 (:goal
  (and
      (contains shot385 cocktail1)
)))
