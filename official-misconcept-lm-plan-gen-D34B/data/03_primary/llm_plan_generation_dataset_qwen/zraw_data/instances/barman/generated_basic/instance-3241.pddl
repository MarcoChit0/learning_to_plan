(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot258 - shot
      ingredient9 ingredient391 ingredient317 - ingredient
      cocktail380 - cocktail
      dispenser178 dispenser303 dispenser219 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot258)
  (dispenses dispenser178 ingredient9)
  (dispenses dispenser303 ingredient391)
  (dispenses dispenser219 ingredient317)
  (clean shaker428)
  (clean shot258)
  (empty shaker428)
  (empty shot258)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail380 ingredient391)
  (cocktail-part2 cocktail380 ingredient317)
)
 (:goal
  (and
      (contains shot258 cocktail380)
)))
