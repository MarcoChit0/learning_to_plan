(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot181 shot150 - shot
      ingredient181 ingredient97 ingredient282 ingredient207 - ingredient
      cocktail1 - cocktail
      dispenser451 dispenser99 dispenser280 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot181)
  (ontable shot150)
  (dispenses dispenser451 ingredient181)
  (dispenses dispenser99 ingredient97)
  (dispenses dispenser280 ingredient282)
  (dispenses dispenser331 ingredient207)
  (clean shaker388)
  (clean shot181)
  (clean shot150)
  (empty shaker388)
  (empty shot181)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient207)
  (cocktail-part2 cocktail1 ingredient181)
)
 (:goal
  (and
      (contains shot181 cocktail1)
)))
