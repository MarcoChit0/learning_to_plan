(define (problem prob)
 (:domain barman)
 (:objects 
      shaker153 - shaker
      left right - hand
      shot404 shot199 - shot
      ingredient85 ingredient194 ingredient492 - ingredient
      cocktail78 - cocktail
      dispenser186 dispenser9 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker153)
  (ontable shot404)
  (ontable shot199)
  (dispenses dispenser186 ingredient85)
  (dispenses dispenser9 ingredient194)
  (dispenses dispenser220 ingredient492)
  (clean shaker153)
  (clean shot404)
  (clean shot199)
  (empty shaker153)
  (empty shot404)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker153 l0)
  (shaker-level shaker153 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail78 ingredient194)
  (cocktail-part2 cocktail78 ingredient492)
)
 (:goal
  (and
      (contains shot404 cocktail78)
)))
