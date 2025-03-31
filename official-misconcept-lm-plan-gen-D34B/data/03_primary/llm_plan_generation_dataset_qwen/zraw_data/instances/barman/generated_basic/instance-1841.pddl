(define (problem prob)
 (:domain barman)
 (:objects 
      shaker33 - shaker
      left right - hand
      shot437 shot242 - shot
      ingredient379 ingredient51 ingredient156 - ingredient
      cocktail127 - cocktail
      dispenser402 dispenser316 dispenser325 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker33)
  (ontable shot437)
  (ontable shot242)
  (dispenses dispenser402 ingredient379)
  (dispenses dispenser316 ingredient51)
  (dispenses dispenser325 ingredient156)
  (clean shaker33)
  (clean shot437)
  (clean shot242)
  (empty shaker33)
  (empty shot437)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker33 l0)
  (shaker-level shaker33 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient156)
  (cocktail-part2 cocktail127 ingredient379)
)
 (:goal
  (and
      (contains shot437 cocktail127)
)))
