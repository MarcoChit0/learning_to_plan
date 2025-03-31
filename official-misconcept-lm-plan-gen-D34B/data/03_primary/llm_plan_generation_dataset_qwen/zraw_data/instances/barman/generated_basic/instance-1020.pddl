(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot48 - shot
      ingredient489 ingredient473 ingredient137 - ingredient
      cocktail1 - cocktail
      dispenser450 dispenser394 dispenser228 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot48)
  (dispenses dispenser450 ingredient489)
  (dispenses dispenser394 ingredient473)
  (dispenses dispenser228 ingredient137)
  (clean shaker109)
  (clean shot48)
  (empty shaker109)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient137)
  (cocktail-part2 cocktail1 ingredient473)
)
 (:goal
  (and
      (contains shot48 cocktail1)
)))
