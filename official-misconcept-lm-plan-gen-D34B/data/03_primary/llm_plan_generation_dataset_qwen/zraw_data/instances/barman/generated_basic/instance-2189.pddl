(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot490 shot189 shot181 - shot
      ingredient281 ingredient205 - ingredient
      cocktail211 - cocktail
      dispenser392 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot490)
  (ontable shot189)
  (ontable shot181)
  (dispenses dispenser392 ingredient281)
  (dispenses dispenser296 ingredient205)
  (clean shaker432)
  (clean shot490)
  (clean shot189)
  (clean shot181)
  (empty shaker432)
  (empty shot490)
  (empty shot189)
  (empty shot181)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail211 ingredient281)
  (cocktail-part2 cocktail211 ingredient205)
)
 (:goal
  (and
      (contains shot490 cocktail211)
      (contains shot189 cocktail211)
)))
