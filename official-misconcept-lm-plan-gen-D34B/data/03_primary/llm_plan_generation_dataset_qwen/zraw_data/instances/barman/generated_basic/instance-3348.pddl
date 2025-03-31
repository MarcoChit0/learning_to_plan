(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot184 shot459 - shot
      ingredient205 ingredient346 - ingredient
      cocktail326 - cocktail
      dispenser177 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot184)
  (ontable shot459)
  (dispenses dispenser177 ingredient205)
  (dispenses dispenser250 ingredient346)
  (clean shaker460)
  (clean shot184)
  (clean shot459)
  (empty shaker460)
  (empty shot184)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient346)
  (cocktail-part2 cocktail326 ingredient205)
)
 (:goal
  (and
      (contains shot184 cocktail326)
)))
