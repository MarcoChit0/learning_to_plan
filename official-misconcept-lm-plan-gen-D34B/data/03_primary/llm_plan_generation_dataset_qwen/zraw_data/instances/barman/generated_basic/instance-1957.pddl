(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot221 shot368 shot280 - shot
      ingredient60 ingredient310 ingredient132 ingredient498 - ingredient
      cocktail35 - cocktail
      dispenser98 dispenser426 dispenser272 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot221)
  (ontable shot368)
  (ontable shot280)
  (dispenses dispenser98 ingredient60)
  (dispenses dispenser426 ingredient310)
  (dispenses dispenser272 ingredient132)
  (dispenses dispenser151 ingredient498)
  (clean shaker398)
  (clean shot221)
  (clean shot368)
  (clean shot280)
  (empty shaker398)
  (empty shot221)
  (empty shot368)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail35 ingredient310)
  (cocktail-part2 cocktail35 ingredient132)
)
 (:goal
  (and
      (contains shot221 cocktail35)
      (contains shot368 ingredient132)
)))
