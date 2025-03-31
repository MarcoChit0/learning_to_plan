(define (problem prob)
 (:domain barman)
 (:objects 
      shaker131 - shaker
      left right - hand
      shot473 shot326 - shot
      ingredient119 ingredient86 ingredient479 ingredient304 - ingredient
      cocktail374 - cocktail
      dispenser74 dispenser32 dispenser159 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker131)
  (ontable shot473)
  (ontable shot326)
  (dispenses dispenser74 ingredient119)
  (dispenses dispenser32 ingredient86)
  (dispenses dispenser159 ingredient479)
  (dispenses dispenser59 ingredient304)
  (clean shaker131)
  (clean shot473)
  (clean shot326)
  (empty shaker131)
  (empty shot473)
  (empty shot326)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker131 l0)
  (shaker-level shaker131 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail374 ingredient119)
  (cocktail-part2 cocktail374 ingredient479)
)
 (:goal
  (and
      (contains shot473 cocktail374)
)))
