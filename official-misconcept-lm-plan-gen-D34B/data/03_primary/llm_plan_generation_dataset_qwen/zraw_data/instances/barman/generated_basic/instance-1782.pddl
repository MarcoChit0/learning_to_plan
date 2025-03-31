(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot133 shot197 - shot
      ingredient265 ingredient447 ingredient308 - ingredient
      cocktail39 - cocktail
      dispenser443 dispenser318 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot133)
  (ontable shot197)
  (dispenses dispenser443 ingredient265)
  (dispenses dispenser318 ingredient447)
  (dispenses dispenser322 ingredient308)
  (clean shaker63)
  (clean shot133)
  (clean shot197)
  (empty shaker63)
  (empty shot133)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient308)
  (cocktail-part2 cocktail39 ingredient447)
)
 (:goal
  (and
      (contains shot133 cocktail39)
)))
