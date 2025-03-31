(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot164 - shot
      ingredient296 ingredient204 ingredient376 ingredient82 - ingredient
      cocktail339 - cocktail
      dispenser305 dispenser325 dispenser242 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot164)
  (dispenses dispenser305 ingredient296)
  (dispenses dispenser325 ingredient204)
  (dispenses dispenser242 ingredient376)
  (dispenses dispenser205 ingredient82)
  (clean shaker116)
  (clean shot164)
  (empty shaker116)
  (empty shot164)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail339 ingredient296)
  (cocktail-part2 cocktail339 ingredient204)
)
 (:goal
  (and
      (contains shot164 cocktail339)
)))
