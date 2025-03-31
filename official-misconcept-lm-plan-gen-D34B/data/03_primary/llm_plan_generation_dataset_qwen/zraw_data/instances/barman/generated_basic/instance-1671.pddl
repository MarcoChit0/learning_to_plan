(define (problem prob)
 (:domain barman)
 (:objects 
      shaker415 - shaker
      left right - hand
      shot456 shot112 - shot
      ingredient344 ingredient271 ingredient457 ingredient252 - ingredient
      cocktail323 - cocktail
      dispenser456 dispenser73 dispenser179 dispenser108 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker415)
  (ontable shot456)
  (ontable shot112)
  (dispenses dispenser456 ingredient344)
  (dispenses dispenser73 ingredient271)
  (dispenses dispenser179 ingredient457)
  (dispenses dispenser108 ingredient252)
  (clean shaker415)
  (clean shot456)
  (clean shot112)
  (empty shaker415)
  (empty shot456)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker415 l0)
  (shaker-level shaker415 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail323 ingredient252)
  (cocktail-part2 cocktail323 ingredient271)
)
 (:goal
  (and
      (contains shot456 cocktail323)
)))
