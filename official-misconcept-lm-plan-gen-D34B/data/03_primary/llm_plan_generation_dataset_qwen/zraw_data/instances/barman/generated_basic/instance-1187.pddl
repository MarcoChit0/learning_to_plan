(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot439 shot353 - shot
      ingredient43 ingredient182 - ingredient
      cocktail409 - cocktail
      dispenser209 dispenser499 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot439)
  (ontable shot353)
  (dispenses dispenser209 ingredient43)
  (dispenses dispenser499 ingredient182)
  (clean shaker462)
  (clean shot439)
  (clean shot353)
  (empty shaker462)
  (empty shot439)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail409 ingredient43)
  (cocktail-part2 cocktail409 ingredient182)
)
 (:goal
  (and
      (contains shot439 cocktail409)
)))
