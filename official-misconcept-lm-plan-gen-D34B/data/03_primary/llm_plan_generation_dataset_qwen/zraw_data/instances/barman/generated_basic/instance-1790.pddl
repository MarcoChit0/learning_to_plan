(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot394 shot395 - shot
      ingredient156 ingredient343 ingredient461 - ingredient
      cocktail433 - cocktail
      dispenser126 dispenser196 dispenser103 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot394)
  (ontable shot395)
  (dispenses dispenser126 ingredient156)
  (dispenses dispenser196 ingredient343)
  (dispenses dispenser103 ingredient461)
  (clean shaker266)
  (clean shot394)
  (clean shot395)
  (empty shaker266)
  (empty shot394)
  (empty shot395)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail433 ingredient156)
  (cocktail-part2 cocktail433 ingredient343)
)
 (:goal
  (and
      (contains shot394 cocktail433)
)))
