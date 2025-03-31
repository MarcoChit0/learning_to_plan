(define (problem prob)
 (:domain barman)
 (:objects 
      shaker150 - shaker
      left right - hand
      shot63 shot221 shot394 - shot
      ingredient89 ingredient412 ingredient54 - ingredient
      cocktail66 - cocktail
      dispenser364 dispenser43 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker150)
  (ontable shot63)
  (ontable shot221)
  (ontable shot394)
  (dispenses dispenser364 ingredient89)
  (dispenses dispenser43 ingredient412)
  (dispenses dispenser402 ingredient54)
  (clean shaker150)
  (clean shot63)
  (clean shot221)
  (clean shot394)
  (empty shaker150)
  (empty shot63)
  (empty shot221)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker150 l0)
  (shaker-level shaker150 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail66 ingredient89)
  (cocktail-part2 cocktail66 ingredient412)
)
 (:goal
  (and
      (contains shot63 cocktail66)
      (contains shot221 ingredient54)
)))
