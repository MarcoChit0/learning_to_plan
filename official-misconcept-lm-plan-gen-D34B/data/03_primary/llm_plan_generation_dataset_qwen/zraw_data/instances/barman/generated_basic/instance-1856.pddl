(define (problem prob)
 (:domain barman)
 (:objects 
      shaker81 - shaker
      left right - hand
      shot225 shot12 - shot
      ingredient83 ingredient86 - ingredient
      cocktail343 - cocktail
      dispenser120 dispenser60 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker81)
  (ontable shot225)
  (ontable shot12)
  (dispenses dispenser120 ingredient83)
  (dispenses dispenser60 ingredient86)
  (clean shaker81)
  (clean shot225)
  (clean shot12)
  (empty shaker81)
  (empty shot225)
  (empty shot12)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker81 l0)
  (shaker-level shaker81 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail343 ingredient83)
  (cocktail-part2 cocktail343 ingredient86)
)
 (:goal
  (and
      (contains shot225 cocktail343)
)))
