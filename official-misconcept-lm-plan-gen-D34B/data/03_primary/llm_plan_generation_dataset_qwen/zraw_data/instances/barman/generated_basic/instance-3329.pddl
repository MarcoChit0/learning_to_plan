(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot69 shot237 shot52 - shot
      ingredient224 ingredient142 ingredient274 - ingredient
      cocktail76 - cocktail
      dispenser39 dispenser439 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot69)
  (ontable shot237)
  (ontable shot52)
  (dispenses dispenser39 ingredient224)
  (dispenses dispenser439 ingredient142)
  (dispenses dispenser306 ingredient274)
  (clean shaker237)
  (clean shot69)
  (clean shot237)
  (clean shot52)
  (empty shaker237)
  (empty shot69)
  (empty shot237)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail76 ingredient274)
  (cocktail-part2 cocktail76 ingredient224)
)
 (:goal
  (and
      (contains shot69 cocktail76)
      (contains shot237 cocktail76)
)))
