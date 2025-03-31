(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot199 shot282 shot65 - shot
      ingredient368 ingredient312 - ingredient
      cocktail448 - cocktail
      dispenser376 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot199)
  (ontable shot282)
  (ontable shot65)
  (dispenses dispenser376 ingredient368)
  (dispenses dispenser43 ingredient312)
  (clean shaker469)
  (clean shot199)
  (clean shot282)
  (clean shot65)
  (empty shaker469)
  (empty shot199)
  (empty shot282)
  (empty shot65)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail448 ingredient312)
  (cocktail-part2 cocktail448 ingredient368)
)
 (:goal
  (and
      (contains shot199 cocktail448)
      (contains shot282 cocktail448)
)))
