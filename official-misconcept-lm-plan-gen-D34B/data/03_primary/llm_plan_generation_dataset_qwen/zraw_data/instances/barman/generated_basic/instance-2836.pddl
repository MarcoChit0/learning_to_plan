(define (problem prob)
 (:domain barman)
 (:objects 
      shaker214 - shaker
      left right - hand
      shot15 shot141 - shot
      ingredient23 ingredient4 ingredient460 ingredient424 - ingredient
      cocktail260 - cocktail
      dispenser295 dispenser304 dispenser426 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker214)
  (ontable shot15)
  (ontable shot141)
  (dispenses dispenser295 ingredient23)
  (dispenses dispenser304 ingredient4)
  (dispenses dispenser426 ingredient460)
  (dispenses dispenser240 ingredient424)
  (clean shaker214)
  (clean shot15)
  (clean shot141)
  (empty shaker214)
  (empty shot15)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker214 l0)
  (shaker-level shaker214 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient424)
  (cocktail-part2 cocktail260 ingredient460)
)
 (:goal
  (and
      (contains shot15 cocktail260)
)))
