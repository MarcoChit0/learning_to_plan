(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot21 - shot
      ingredient105 ingredient125 ingredient80 ingredient211 - ingredient
      cocktail174 - cocktail
      dispenser453 dispenser180 dispenser57 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot21)
  (dispenses dispenser453 ingredient105)
  (dispenses dispenser180 ingredient125)
  (dispenses dispenser57 ingredient80)
  (dispenses dispenser405 ingredient211)
  (clean shaker56)
  (clean shot21)
  (empty shaker56)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient105)
  (cocktail-part2 cocktail174 ingredient125)
)
 (:goal
  (and
      (contains shot21 cocktail174)
)))
