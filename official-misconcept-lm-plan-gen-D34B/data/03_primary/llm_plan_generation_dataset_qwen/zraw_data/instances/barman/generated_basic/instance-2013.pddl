(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot302 shot131 shot104 - shot
      ingredient42 ingredient165 ingredient143 - ingredient
      cocktail446 - cocktail
      dispenser398 dispenser282 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot302)
  (ontable shot131)
  (ontable shot104)
  (dispenses dispenser398 ingredient42)
  (dispenses dispenser282 ingredient165)
  (dispenses dispenser45 ingredient143)
  (clean shaker469)
  (clean shot302)
  (clean shot131)
  (clean shot104)
  (empty shaker469)
  (empty shot302)
  (empty shot131)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail446 ingredient42)
  (cocktail-part2 cocktail446 ingredient165)
)
 (:goal
  (and
      (contains shot302 cocktail446)
      (contains shot131 cocktail446)
)))
