(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot288 shot162 shot363 - shot
      ingredient271 ingredient209 ingredient444 ingredient149 - ingredient
      cocktail118 - cocktail
      dispenser161 dispenser300 dispenser372 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot288)
  (ontable shot162)
  (ontable shot363)
  (dispenses dispenser161 ingredient271)
  (dispenses dispenser300 ingredient209)
  (dispenses dispenser372 ingredient444)
  (dispenses dispenser221 ingredient149)
  (clean shaker297)
  (clean shot288)
  (clean shot162)
  (clean shot363)
  (empty shaker297)
  (empty shot288)
  (empty shot162)
  (empty shot363)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail118 ingredient271)
  (cocktail-part2 cocktail118 ingredient444)
)
 (:goal
  (and
      (contains shot288 cocktail118)
      (contains shot162 ingredient149)
)))
