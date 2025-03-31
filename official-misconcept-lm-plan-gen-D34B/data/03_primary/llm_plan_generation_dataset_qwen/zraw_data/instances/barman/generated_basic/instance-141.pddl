(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot265 - shot
      ingredient171 ingredient302 ingredient204 ingredient46 - ingredient
      cocktail1 - cocktail
      dispenser442 dispenser412 dispenser326 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot265)
  (dispenses dispenser442 ingredient171)
  (dispenses dispenser412 ingredient302)
  (dispenses dispenser326 ingredient204)
  (dispenses dispenser299 ingredient46)
  (clean shaker411)
  (clean shot265)
  (empty shaker411)
  (empty shot265)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient204)
  (cocktail-part2 cocktail1 ingredient302)
)
 (:goal
  (and
      (contains shot265 cocktail1)
)))
