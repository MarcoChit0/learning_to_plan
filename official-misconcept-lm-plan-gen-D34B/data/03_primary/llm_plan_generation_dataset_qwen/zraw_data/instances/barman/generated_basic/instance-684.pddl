(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot39 shot14 - shot
      ingredient191 ingredient442 - ingredient
      cocktail1 - cocktail
      dispenser211 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot39)
  (ontable shot14)
  (dispenses dispenser211 ingredient191)
  (dispenses dispenser278 ingredient442)
  (clean shaker469)
  (clean shot39)
  (clean shot14)
  (empty shaker469)
  (empty shot39)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient191)
  (cocktail-part2 cocktail1 ingredient442)
)
 (:goal
  (and
      (contains shot39 cocktail1)
)))
