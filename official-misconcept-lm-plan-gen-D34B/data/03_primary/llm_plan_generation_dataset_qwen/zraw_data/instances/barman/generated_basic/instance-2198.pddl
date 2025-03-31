(define (problem prob)
 (:domain barman)
 (:objects 
      shaker374 - shaker
      left right - hand
      shot85 shot481 shot141 - shot
      ingredient52 ingredient205 - ingredient
      cocktail265 - cocktail
      dispenser52 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker374)
  (ontable shot85)
  (ontable shot481)
  (ontable shot141)
  (dispenses dispenser52 ingredient52)
  (dispenses dispenser209 ingredient205)
  (clean shaker374)
  (clean shot85)
  (clean shot481)
  (clean shot141)
  (empty shaker374)
  (empty shot85)
  (empty shot481)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker374 l0)
  (shaker-level shaker374 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail265 ingredient205)
  (cocktail-part2 cocktail265 ingredient52)
)
 (:goal
  (and
      (contains shot85 cocktail265)
      (contains shot481 ingredient52)
)))
