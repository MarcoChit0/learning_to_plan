(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot349 - shot
      ingredient381 ingredient345 ingredient293 ingredient445 - ingredient
      cocktail289 - cocktail
      dispenser227 dispenser296 dispenser345 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot349)
  (dispenses dispenser227 ingredient381)
  (dispenses dispenser296 ingredient345)
  (dispenses dispenser345 ingredient293)
  (dispenses dispenser206 ingredient445)
  (clean shaker192)
  (clean shot349)
  (empty shaker192)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail289 ingredient445)
  (cocktail-part2 cocktail289 ingredient381)
)
 (:goal
  (and
      (contains shot349 cocktail289)
)))
