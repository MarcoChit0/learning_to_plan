(define (problem prob)
 (:domain barman)
 (:objects 
      shaker168 - shaker
      left right - hand
      shot3 shot70 shot72 - shot
      ingredient437 ingredient67 ingredient134 - ingredient
      cocktail87 - cocktail
      dispenser188 dispenser254 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker168)
  (ontable shot3)
  (ontable shot70)
  (ontable shot72)
  (dispenses dispenser188 ingredient437)
  (dispenses dispenser254 ingredient67)
  (dispenses dispenser292 ingredient134)
  (clean shaker168)
  (clean shot3)
  (clean shot70)
  (clean shot72)
  (empty shaker168)
  (empty shot3)
  (empty shot70)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker168 l0)
  (shaker-level shaker168 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail87 ingredient437)
  (cocktail-part2 cocktail87 ingredient134)
)
 (:goal
  (and
      (contains shot3 cocktail87)
      (contains shot70 cocktail87)
)))
