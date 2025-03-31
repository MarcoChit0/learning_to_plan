(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot196 shot225 - shot
      ingredient58 ingredient26 ingredient269 - ingredient
      cocktail1 - cocktail
      dispenser411 dispenser153 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot196)
  (ontable shot225)
  (dispenses dispenser411 ingredient58)
  (dispenses dispenser153 ingredient26)
  (dispenses dispenser422 ingredient269)
  (clean shaker266)
  (clean shot196)
  (clean shot225)
  (empty shaker266)
  (empty shot196)
  (empty shot225)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient269)
  (cocktail-part2 cocktail1 ingredient58)
)
 (:goal
  (and
      (contains shot196 cocktail1)
)))
