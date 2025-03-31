(define (problem prob)
 (:domain barman)
 (:objects 
      shaker141 - shaker
      left right - hand
      shot462 shot38 - shot
      ingredient36 ingredient9 ingredient314 - ingredient
      cocktail99 - cocktail
      dispenser420 dispenser119 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker141)
  (ontable shot462)
  (ontable shot38)
  (dispenses dispenser420 ingredient36)
  (dispenses dispenser119 ingredient9)
  (dispenses dispenser69 ingredient314)
  (clean shaker141)
  (clean shot462)
  (clean shot38)
  (empty shaker141)
  (empty shot462)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker141 l0)
  (shaker-level shaker141 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail99 ingredient314)
  (cocktail-part2 cocktail99 ingredient9)
)
 (:goal
  (and
      (contains shot462 cocktail99)
)))
