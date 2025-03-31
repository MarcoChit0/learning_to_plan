(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot67 shot112 - shot
      ingredient12 ingredient360 ingredient225 - ingredient
      cocktail1 - cocktail
      dispenser417 dispenser1 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot67)
  (ontable shot112)
  (dispenses dispenser417 ingredient12)
  (dispenses dispenser1 ingredient360)
  (dispenses dispenser252 ingredient225)
  (clean shaker95)
  (clean shot67)
  (clean shot112)
  (empty shaker95)
  (empty shot67)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient360)
  (cocktail-part2 cocktail1 ingredient12)
)
 (:goal
  (and
      (contains shot67 cocktail1)
)))
