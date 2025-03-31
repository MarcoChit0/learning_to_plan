(define (problem prob)
 (:domain barman)
 (:objects 
      shaker47 - shaker
      left right - hand
      shot331 - shot
      ingredient373 ingredient73 ingredient125 ingredient149 - ingredient
      cocktail188 - cocktail
      dispenser72 dispenser241 dispenser47 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker47)
  (ontable shot331)
  (dispenses dispenser72 ingredient373)
  (dispenses dispenser241 ingredient73)
  (dispenses dispenser47 ingredient125)
  (dispenses dispenser363 ingredient149)
  (clean shaker47)
  (clean shot331)
  (empty shaker47)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker47 l0)
  (shaker-level shaker47 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail188 ingredient149)
  (cocktail-part2 cocktail188 ingredient373)
)
 (:goal
  (and
      (contains shot331 cocktail188)
)))
