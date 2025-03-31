(define (problem prob)
 (:domain barman)
 (:objects 
      shaker421 - shaker
      left right - hand
      shot1 - shot
      ingredient302 ingredient219 ingredient227 - ingredient
      cocktail145 - cocktail
      dispenser244 dispenser153 dispenser241 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker421)
  (ontable shot1)
  (dispenses dispenser244 ingredient302)
  (dispenses dispenser153 ingredient219)
  (dispenses dispenser241 ingredient227)
  (clean shaker421)
  (clean shot1)
  (empty shaker421)
  (empty shot1)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker421 l0)
  (shaker-level shaker421 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail145 ingredient227)
  (cocktail-part2 cocktail145 ingredient219)
)
 (:goal
  (and
      (contains shot1 cocktail145)
)))
