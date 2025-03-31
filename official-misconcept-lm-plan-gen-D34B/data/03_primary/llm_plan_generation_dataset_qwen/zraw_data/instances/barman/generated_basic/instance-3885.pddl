(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot207 shot429 - shot
      ingredient428 ingredient274 ingredient129 - ingredient
      cocktail80 - cocktail
      dispenser409 dispenser171 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot207)
  (ontable shot429)
  (dispenses dispenser409 ingredient428)
  (dispenses dispenser171 ingredient274)
  (dispenses dispenser151 ingredient129)
  (clean shaker152)
  (clean shot207)
  (clean shot429)
  (empty shaker152)
  (empty shot207)
  (empty shot429)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail80 ingredient428)
  (cocktail-part2 cocktail80 ingredient274)
)
 (:goal
  (and
      (contains shot207 cocktail80)
)))
