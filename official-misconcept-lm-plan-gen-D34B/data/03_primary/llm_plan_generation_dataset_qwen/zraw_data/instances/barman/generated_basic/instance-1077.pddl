(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot177 shot295 shot34 - shot
      ingredient38 ingredient276 ingredient165 ingredient230 - ingredient
      cocktail1 - cocktail
      dispenser119 dispenser267 dispenser153 dispenser353 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot177)
  (ontable shot295)
  (ontable shot34)
  (dispenses dispenser119 ingredient38)
  (dispenses dispenser267 ingredient276)
  (dispenses dispenser153 ingredient165)
  (dispenses dispenser353 ingredient230)
  (clean shaker296)
  (clean shot177)
  (clean shot295)
  (clean shot34)
  (empty shaker296)
  (empty shot177)
  (empty shot295)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient38)
  (cocktail-part2 cocktail1 ingredient276)
)
 (:goal
  (and
      (contains shot177 cocktail1)
      (contains shot295 cocktail1)
)))
