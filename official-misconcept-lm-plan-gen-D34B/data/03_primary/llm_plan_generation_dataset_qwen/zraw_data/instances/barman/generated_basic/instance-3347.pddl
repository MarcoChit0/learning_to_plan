(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot410 shot10 - shot
      ingredient52 ingredient422 - ingredient
      cocktail52 - cocktail
      dispenser152 dispenser382 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot410)
  (ontable shot10)
  (dispenses dispenser152 ingredient52)
  (dispenses dispenser382 ingredient422)
  (clean shaker104)
  (clean shot410)
  (clean shot10)
  (empty shaker104)
  (empty shot410)
  (empty shot10)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail52 ingredient52)
  (cocktail-part2 cocktail52 ingredient422)
)
 (:goal
  (and
      (contains shot410 cocktail52)
)))
