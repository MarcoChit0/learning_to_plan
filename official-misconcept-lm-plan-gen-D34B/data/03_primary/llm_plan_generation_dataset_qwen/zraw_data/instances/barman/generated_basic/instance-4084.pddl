(define (problem prob)
 (:domain barman)
 (:objects 
      shaker343 - shaker
      left right - hand
      shot60 shot19 - shot
      ingredient91 ingredient17 ingredient49 ingredient99 - ingredient
      cocktail448 - cocktail
      dispenser336 dispenser202 dispenser217 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker343)
  (ontable shot60)
  (ontable shot19)
  (dispenses dispenser336 ingredient91)
  (dispenses dispenser202 ingredient17)
  (dispenses dispenser217 ingredient49)
  (dispenses dispenser428 ingredient99)
  (clean shaker343)
  (clean shot60)
  (clean shot19)
  (empty shaker343)
  (empty shot60)
  (empty shot19)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker343 l0)
  (shaker-level shaker343 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail448 ingredient17)
  (cocktail-part2 cocktail448 ingredient91)
)
 (:goal
  (and
      (contains shot60 cocktail448)
)))
