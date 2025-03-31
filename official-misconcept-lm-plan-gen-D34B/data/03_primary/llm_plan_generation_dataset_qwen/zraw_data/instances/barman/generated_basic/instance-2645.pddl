(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot249 shot151 - shot
      ingredient274 ingredient186 ingredient133 - ingredient
      cocktail487 - cocktail
      dispenser186 dispenser15 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot249)
  (ontable shot151)
  (dispenses dispenser186 ingredient274)
  (dispenses dispenser15 ingredient186)
  (dispenses dispenser363 ingredient133)
  (clean shaker326)
  (clean shot249)
  (clean shot151)
  (empty shaker326)
  (empty shot249)
  (empty shot151)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail487 ingredient274)
  (cocktail-part2 cocktail487 ingredient133)
)
 (:goal
  (and
      (contains shot249 cocktail487)
)))
