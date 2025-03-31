(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot353 shot416 - shot
      ingredient410 ingredient47 ingredient303 ingredient465 - ingredient
      cocktail357 - cocktail
      dispenser253 dispenser256 dispenser177 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot353)
  (ontable shot416)
  (dispenses dispenser253 ingredient410)
  (dispenses dispenser256 ingredient47)
  (dispenses dispenser177 ingredient303)
  (dispenses dispenser418 ingredient465)
  (clean shaker436)
  (clean shot353)
  (clean shot416)
  (empty shaker436)
  (empty shot353)
  (empty shot416)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail357 ingredient303)
  (cocktail-part2 cocktail357 ingredient47)
)
 (:goal
  (and
      (contains shot353 cocktail357)
)))
