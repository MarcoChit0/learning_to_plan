(define (problem prob)
 (:domain barman)
 (:objects 
      shaker218 - shaker
      left right - hand
      shot397 shot297 - shot
      ingredient289 ingredient350 ingredient240 - ingredient
      cocktail241 - cocktail
      dispenser85 dispenser435 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker218)
  (ontable shot397)
  (ontable shot297)
  (dispenses dispenser85 ingredient289)
  (dispenses dispenser435 ingredient350)
  (dispenses dispenser474 ingredient240)
  (clean shaker218)
  (clean shot397)
  (clean shot297)
  (empty shaker218)
  (empty shot397)
  (empty shot297)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker218 l0)
  (shaker-level shaker218 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient289)
  (cocktail-part2 cocktail241 ingredient240)
)
 (:goal
  (and
      (contains shot397 cocktail241)
)))
