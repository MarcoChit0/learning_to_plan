(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot298 shot205 shot120 - shot
      ingredient147 ingredient448 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser428 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot298)
  (ontable shot205)
  (ontable shot120)
  (dispenses dispenser428 ingredient147)
  (dispenses dispenser417 ingredient448)
  (clean shaker479)
  (clean shot298)
  (clean shot205)
  (clean shot120)
  (empty shaker479)
  (empty shot298)
  (empty shot205)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient147)
  (cocktail-part2 cocktail1 ingredient448)
  (cocktail-part1 cocktail2 ingredient448)
  (cocktail-part2 cocktail2 ingredient147)
)
 (:goal
  (and
      (contains shot298 cocktail2)
      (contains shot205 cocktail1)
)))
