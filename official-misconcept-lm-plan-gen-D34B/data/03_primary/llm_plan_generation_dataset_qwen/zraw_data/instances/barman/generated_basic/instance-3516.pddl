(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot60 shot491 - shot
      ingredient255 ingredient220 ingredient454 ingredient208 - ingredient
      cocktail260 - cocktail
      dispenser273 dispenser157 dispenser197 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot60)
  (ontable shot491)
  (dispenses dispenser273 ingredient255)
  (dispenses dispenser157 ingredient220)
  (dispenses dispenser197 ingredient454)
  (dispenses dispenser65 ingredient208)
  (clean shaker329)
  (clean shot60)
  (clean shot491)
  (empty shaker329)
  (empty shot60)
  (empty shot491)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient454)
  (cocktail-part2 cocktail260 ingredient255)
)
 (:goal
  (and
      (contains shot60 cocktail260)
)))
