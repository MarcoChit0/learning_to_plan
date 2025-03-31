(define (problem prob)
 (:domain barman)
 (:objects 
      shaker482 - shaker
      left right - hand
      shot130 shot20 - shot
      ingredient7 ingredient478 ingredient72 ingredient223 - ingredient
      cocktail482 - cocktail
      dispenser175 dispenser337 dispenser113 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker482)
  (ontable shot130)
  (ontable shot20)
  (dispenses dispenser175 ingredient7)
  (dispenses dispenser337 ingredient478)
  (dispenses dispenser113 ingredient72)
  (dispenses dispenser303 ingredient223)
  (clean shaker482)
  (clean shot130)
  (clean shot20)
  (empty shaker482)
  (empty shot130)
  (empty shot20)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker482 l0)
  (shaker-level shaker482 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient478)
  (cocktail-part2 cocktail482 ingredient223)
)
 (:goal
  (and
      (contains shot130 cocktail482)
)))
