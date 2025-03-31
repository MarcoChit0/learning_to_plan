(define (problem prob)
 (:domain barman)
 (:objects 
      shaker79 - shaker
      left right - hand
      shot492 - shot
      ingredient424 ingredient7 ingredient15 - ingredient
      cocktail1 - cocktail
      dispenser264 dispenser24 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker79)
  (ontable shot492)
  (dispenses dispenser264 ingredient424)
  (dispenses dispenser24 ingredient7)
  (dispenses dispenser134 ingredient15)
  (clean shaker79)
  (clean shot492)
  (empty shaker79)
  (empty shot492)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker79 l0)
  (shaker-level shaker79 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient7)
  (cocktail-part2 cocktail1 ingredient15)
)
 (:goal
  (and
      (contains shot492 cocktail1)
)))
