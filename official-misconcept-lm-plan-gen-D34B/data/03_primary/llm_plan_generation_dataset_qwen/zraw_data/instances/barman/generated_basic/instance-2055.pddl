(define (problem prob)
 (:domain barman)
 (:objects 
      shaker94 - shaker
      left right - hand
      shot125 - shot
      ingredient101 ingredient290 ingredient439 ingredient331 - ingredient
      cocktail199 - cocktail
      dispenser260 dispenser1 dispenser271 dispenser241 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker94)
  (ontable shot125)
  (dispenses dispenser260 ingredient101)
  (dispenses dispenser1 ingredient290)
  (dispenses dispenser271 ingredient439)
  (dispenses dispenser241 ingredient331)
  (clean shaker94)
  (clean shot125)
  (empty shaker94)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker94 l0)
  (shaker-level shaker94 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail199 ingredient439)
  (cocktail-part2 cocktail199 ingredient290)
)
 (:goal
  (and
      (contains shot125 cocktail199)
)))
