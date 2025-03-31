(define (problem prob)
 (:domain barman)
 (:objects 
      shaker108 - shaker
      left right - hand
      shot22 - shot
      ingredient340 ingredient436 ingredient92 ingredient15 - ingredient
      cocktail84 - cocktail
      dispenser278 dispenser441 dispenser230 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker108)
  (ontable shot22)
  (dispenses dispenser278 ingredient340)
  (dispenses dispenser441 ingredient436)
  (dispenses dispenser230 ingredient92)
  (dispenses dispenser425 ingredient15)
  (clean shaker108)
  (clean shot22)
  (empty shaker108)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker108 l0)
  (shaker-level shaker108 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail84 ingredient15)
  (cocktail-part2 cocktail84 ingredient92)
)
 (:goal
  (and
      (contains shot22 cocktail84)
)))
