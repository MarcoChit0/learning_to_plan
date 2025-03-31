(define (problem prob)
 (:domain barman)
 (:objects 
      shaker331 - shaker
      left right - hand
      shot423 - shot
      ingredient32 ingredient123 ingredient479 ingredient391 - ingredient
      cocktail91 - cocktail
      dispenser285 dispenser357 dispenser227 dispenser183 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker331)
  (ontable shot423)
  (dispenses dispenser285 ingredient32)
  (dispenses dispenser357 ingredient123)
  (dispenses dispenser227 ingredient479)
  (dispenses dispenser183 ingredient391)
  (clean shaker331)
  (clean shot423)
  (empty shaker331)
  (empty shot423)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker331 l0)
  (shaker-level shaker331 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail91 ingredient479)
  (cocktail-part2 cocktail91 ingredient32)
)
 (:goal
  (and
      (contains shot423 cocktail91)
)))
