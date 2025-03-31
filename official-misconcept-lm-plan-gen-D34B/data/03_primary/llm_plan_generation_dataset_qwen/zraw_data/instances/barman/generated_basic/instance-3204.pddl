(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot268 shot289 - shot
      ingredient442 ingredient366 ingredient496 - ingredient
      cocktail218 - cocktail
      dispenser321 dispenser87 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot268)
  (ontable shot289)
  (dispenses dispenser321 ingredient442)
  (dispenses dispenser87 ingredient366)
  (dispenses dispenser385 ingredient496)
  (clean shaker373)
  (clean shot268)
  (clean shot289)
  (empty shaker373)
  (empty shot268)
  (empty shot289)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail218 ingredient366)
  (cocktail-part2 cocktail218 ingredient442)
)
 (:goal
  (and
      (contains shot268 cocktail218)
)))
