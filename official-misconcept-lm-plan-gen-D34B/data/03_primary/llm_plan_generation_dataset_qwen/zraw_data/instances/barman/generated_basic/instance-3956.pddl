(define (problem prob)
 (:domain barman)
 (:objects 
      shaker92 - shaker
      left right - hand
      shot391 shot263 - shot
      ingredient138 ingredient312 ingredient371 ingredient414 - ingredient
      cocktail273 - cocktail
      dispenser24 dispenser285 dispenser156 dispenser174 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker92)
  (ontable shot391)
  (ontable shot263)
  (dispenses dispenser24 ingredient138)
  (dispenses dispenser285 ingredient312)
  (dispenses dispenser156 ingredient371)
  (dispenses dispenser174 ingredient414)
  (clean shaker92)
  (clean shot391)
  (clean shot263)
  (empty shaker92)
  (empty shot391)
  (empty shot263)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker92 l0)
  (shaker-level shaker92 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail273 ingredient414)
  (cocktail-part2 cocktail273 ingredient312)
)
 (:goal
  (and
      (contains shot391 cocktail273)
)))
