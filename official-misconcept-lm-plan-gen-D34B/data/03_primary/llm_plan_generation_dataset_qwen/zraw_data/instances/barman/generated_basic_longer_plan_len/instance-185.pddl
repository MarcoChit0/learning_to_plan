(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot277 shot446 shot135 - shot
      ingredient149 ingredient384 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser5 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot277)
  (ontable shot446)
  (ontable shot135)
  (dispenses dispenser5 ingredient149)
  (dispenses dispenser346 ingredient384)
  (clean shaker19)
  (clean shot277)
  (clean shot446)
  (clean shot135)
  (empty shaker19)
  (empty shot277)
  (empty shot446)
  (empty shot135)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient384)
  (cocktail-part2 cocktail1 ingredient149)
  (cocktail-part1 cocktail2 ingredient384)
  (cocktail-part2 cocktail2 ingredient149)
)
 (:goal
  (and
      (contains shot277 cocktail2)
      (contains shot446 cocktail1)
)))
