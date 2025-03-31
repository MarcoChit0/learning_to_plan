(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot409 shot434 - shot
      ingredient454 ingredient267 ingredient217 ingredient324 - ingredient
      cocktail1 - cocktail
      dispenser374 dispenser29 dispenser186 dispenser462 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot409)
  (ontable shot434)
  (dispenses dispenser374 ingredient454)
  (dispenses dispenser29 ingredient267)
  (dispenses dispenser186 ingredient217)
  (dispenses dispenser462 ingredient324)
  (clean shaker181)
  (clean shot409)
  (clean shot434)
  (empty shaker181)
  (empty shot409)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient454)
  (cocktail-part2 cocktail1 ingredient324)
)
 (:goal
  (and
      (contains shot409 cocktail1)
)))
