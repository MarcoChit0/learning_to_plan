(define (problem prob)
 (:domain barman)
 (:objects 
      shaker430 - shaker
      left right - hand
      shot476 shot182 - shot
      ingredient181 ingredient137 ingredient271 - ingredient
      cocktail56 - cocktail
      dispenser415 dispenser116 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker430)
  (ontable shot476)
  (ontable shot182)
  (dispenses dispenser415 ingredient181)
  (dispenses dispenser116 ingredient137)
  (dispenses dispenser303 ingredient271)
  (clean shaker430)
  (clean shot476)
  (clean shot182)
  (empty shaker430)
  (empty shot476)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker430 l0)
  (shaker-level shaker430 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient181)
  (cocktail-part2 cocktail56 ingredient271)
)
 (:goal
  (and
      (contains shot476 cocktail56)
)))
