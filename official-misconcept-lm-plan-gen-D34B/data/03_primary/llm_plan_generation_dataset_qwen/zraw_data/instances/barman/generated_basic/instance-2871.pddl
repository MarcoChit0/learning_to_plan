(define (problem prob)
 (:domain barman)
 (:objects 
      shaker230 - shaker
      left right - hand
      shot148 shot68 - shot
      ingredient427 ingredient241 ingredient394 ingredient343 - ingredient
      cocktail441 - cocktail
      dispenser165 dispenser185 dispenser491 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker230)
  (ontable shot148)
  (ontable shot68)
  (dispenses dispenser165 ingredient427)
  (dispenses dispenser185 ingredient241)
  (dispenses dispenser491 ingredient394)
  (dispenses dispenser126 ingredient343)
  (clean shaker230)
  (clean shot148)
  (clean shot68)
  (empty shaker230)
  (empty shot148)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker230 l0)
  (shaker-level shaker230 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail441 ingredient241)
  (cocktail-part2 cocktail441 ingredient343)
)
 (:goal
  (and
      (contains shot148 cocktail441)
)))
