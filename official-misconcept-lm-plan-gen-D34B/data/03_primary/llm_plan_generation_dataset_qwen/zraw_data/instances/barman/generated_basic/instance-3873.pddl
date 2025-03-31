(define (problem prob)
 (:domain barman)
 (:objects 
      shaker205 - shaker
      left right - hand
      shot374 shot174 - shot
      ingredient492 ingredient329 ingredient128 - ingredient
      cocktail163 - cocktail
      dispenser111 dispenser325 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker205)
  (ontable shot374)
  (ontable shot174)
  (dispenses dispenser111 ingredient492)
  (dispenses dispenser325 ingredient329)
  (dispenses dispenser272 ingredient128)
  (clean shaker205)
  (clean shot374)
  (clean shot174)
  (empty shaker205)
  (empty shot374)
  (empty shot174)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker205 l0)
  (shaker-level shaker205 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail163 ingredient329)
  (cocktail-part2 cocktail163 ingredient492)
)
 (:goal
  (and
      (contains shot374 cocktail163)
)))
