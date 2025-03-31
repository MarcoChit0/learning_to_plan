(define (problem prob)
 (:domain barman)
 (:objects 
      shaker367 - shaker
      left right - hand
      shot364 shot193 - shot
      ingredient156 ingredient321 - ingredient
      cocktail326 - cocktail
      dispenser39 dispenser326 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker367)
  (ontable shot364)
  (ontable shot193)
  (dispenses dispenser39 ingredient156)
  (dispenses dispenser326 ingredient321)
  (clean shaker367)
  (clean shot364)
  (clean shot193)
  (empty shaker367)
  (empty shot364)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker367 l0)
  (shaker-level shaker367 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient156)
  (cocktail-part2 cocktail326 ingredient321)
)
 (:goal
  (and
      (contains shot364 cocktail326)
)))
