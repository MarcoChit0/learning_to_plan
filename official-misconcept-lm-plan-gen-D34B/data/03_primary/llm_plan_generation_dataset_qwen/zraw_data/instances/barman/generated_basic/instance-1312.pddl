(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot300 shot269 shot284 shot422 - shot
      ingredient389 ingredient315 - ingredient
      cocktail197 - cocktail
      dispenser335 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot300)
  (ontable shot269)
  (ontable shot284)
  (ontable shot422)
  (dispenses dispenser335 ingredient389)
  (dispenses dispenser466 ingredient315)
  (clean shaker461)
  (clean shot300)
  (clean shot269)
  (clean shot284)
  (clean shot422)
  (empty shaker461)
  (empty shot300)
  (empty shot269)
  (empty shot284)
  (empty shot422)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail197 ingredient389)
  (cocktail-part2 cocktail197 ingredient315)
)
 (:goal
  (and
      (contains shot300 cocktail197)
      (contains shot269 ingredient389)
      (contains shot284 cocktail197)
)))
