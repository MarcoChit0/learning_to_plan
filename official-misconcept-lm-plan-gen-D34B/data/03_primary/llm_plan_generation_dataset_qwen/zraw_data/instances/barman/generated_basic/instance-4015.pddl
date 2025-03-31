(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot386 shot346 - shot
      ingredient141 ingredient97 ingredient132 - ingredient
      cocktail200 - cocktail
      dispenser321 dispenser272 dispenser190 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot386)
  (ontable shot346)
  (dispenses dispenser321 ingredient141)
  (dispenses dispenser272 ingredient97)
  (dispenses dispenser190 ingredient132)
  (clean shaker192)
  (clean shot386)
  (clean shot346)
  (empty shaker192)
  (empty shot386)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient141)
  (cocktail-part2 cocktail200 ingredient97)
)
 (:goal
  (and
      (contains shot386 cocktail200)
)))
