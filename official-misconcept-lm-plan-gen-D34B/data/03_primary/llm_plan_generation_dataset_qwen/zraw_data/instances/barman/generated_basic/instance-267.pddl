(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot436 shot405 shot143 - shot
      ingredient378 ingredient447 ingredient276 - ingredient
      cocktail1 - cocktail
      dispenser248 dispenser216 dispenser315 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot436)
  (ontable shot405)
  (ontable shot143)
  (dispenses dispenser248 ingredient378)
  (dispenses dispenser216 ingredient447)
  (dispenses dispenser315 ingredient276)
  (clean shaker314)
  (clean shot436)
  (clean shot405)
  (clean shot143)
  (empty shaker314)
  (empty shot436)
  (empty shot405)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient447)
  (cocktail-part2 cocktail1 ingredient378)
)
 (:goal
  (and
      (contains shot436 cocktail1)
      (contains shot405 ingredient447)
)))
