(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot54 shot333 shot25 - shot
      ingredient30 ingredient203 ingredient356 ingredient36 - ingredient
      cocktail1 - cocktail
      dispenser16 dispenser297 dispenser223 dispenser251 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot54)
  (ontable shot333)
  (ontable shot25)
  (dispenses dispenser16 ingredient30)
  (dispenses dispenser297 ingredient203)
  (dispenses dispenser223 ingredient356)
  (dispenses dispenser251 ingredient36)
  (clean shaker241)
  (clean shot54)
  (clean shot333)
  (clean shot25)
  (empty shaker241)
  (empty shot54)
  (empty shot333)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient203)
  (cocktail-part2 cocktail1 ingredient30)
)
 (:goal
  (and
      (contains shot54 cocktail1)
      (contains shot333 cocktail1)
)))
