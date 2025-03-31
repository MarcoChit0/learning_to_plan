(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot359 shot268 - shot
      ingredient219 ingredient115 - ingredient
      cocktail410 - cocktail
      dispenser324 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot359)
  (ontable shot268)
  (dispenses dispenser324 ingredient219)
  (dispenses dispenser269 ingredient115)
  (clean shaker91)
  (clean shot359)
  (clean shot268)
  (empty shaker91)
  (empty shot359)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail410 ingredient219)
  (cocktail-part2 cocktail410 ingredient115)
)
 (:goal
  (and
      (contains shot359 cocktail410)
)))
