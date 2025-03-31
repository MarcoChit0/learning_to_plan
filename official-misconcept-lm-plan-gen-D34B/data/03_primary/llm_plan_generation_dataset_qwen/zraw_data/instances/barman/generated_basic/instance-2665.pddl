(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot127 - shot
      ingredient251 ingredient472 ingredient464 ingredient240 - ingredient
      cocktail81 - cocktail
      dispenser129 dispenser341 dispenser300 dispenser81 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot127)
  (dispenses dispenser129 ingredient251)
  (dispenses dispenser341 ingredient472)
  (dispenses dispenser300 ingredient464)
  (dispenses dispenser81 ingredient240)
  (clean shaker278)
  (clean shot127)
  (empty shaker278)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail81 ingredient251)
  (cocktail-part2 cocktail81 ingredient464)
)
 (:goal
  (and
      (contains shot127 cocktail81)
)))
