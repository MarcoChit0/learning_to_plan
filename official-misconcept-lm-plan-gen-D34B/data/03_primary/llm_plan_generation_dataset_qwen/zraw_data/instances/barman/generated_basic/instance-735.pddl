(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot265 shot490 - shot
      ingredient329 ingredient441 ingredient102 - ingredient
      cocktail1 - cocktail
      dispenser350 dispenser193 dispenser260 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot265)
  (ontable shot490)
  (dispenses dispenser350 ingredient329)
  (dispenses dispenser193 ingredient441)
  (dispenses dispenser260 ingredient102)
  (clean shaker438)
  (clean shot265)
  (clean shot490)
  (empty shaker438)
  (empty shot265)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient102)
  (cocktail-part2 cocktail1 ingredient441)
)
 (:goal
  (and
      (contains shot265 cocktail1)
)))
