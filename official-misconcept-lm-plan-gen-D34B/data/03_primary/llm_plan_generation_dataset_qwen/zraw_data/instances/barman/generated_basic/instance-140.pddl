(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot101 - shot
      ingredient176 ingredient122 ingredient152 ingredient449 - ingredient
      cocktail1 - cocktail
      dispenser477 dispenser167 dispenser52 dispenser183 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot101)
  (dispenses dispenser477 ingredient176)
  (dispenses dispenser167 ingredient122)
  (dispenses dispenser52 ingredient152)
  (dispenses dispenser183 ingredient449)
  (clean shaker77)
  (clean shot101)
  (empty shaker77)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient449)
  (cocktail-part2 cocktail1 ingredient152)
)
 (:goal
  (and
      (contains shot101 cocktail1)
)))
